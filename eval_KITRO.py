import argparse
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
import json

# local libraries 
from lib.cores import config 
from lib.models.smpl import SMPL_59, H36M_TO_J14
from lib.models.kitro import KITRO_refine
from lib.utils.eval_utils import batch_compute_similarity_transform_torch, compute_error_verts_torch

class SMPL_Estimates_Dataset(Dataset):
    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the saved .pth file containing the initial SMPL prediction data.
        """
        # Load the data from the saved .pt file
        self.data = torch.load(data_path)
        # Number of samples in the dataset
        self.num_samples = self.data['pred_theta'].shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve the data for the given index
        sample = {
            'imgname': self.data['imagename'][idx],  # image name (list)
            'pred_theta': self.data['pred_theta'][idx],  # Predicted 3D rotation matrix (shape: [samples, 24, 3, 3])
            'pred_beta': self.data['pred_beta'][idx],    # Predicted body shape parameters (shape: [samples, 10])
            'pred_cam': self.data['pred_cam'][idx],      # Predicted camera translation (shape: [samples, 3])
            'intrinsics': self.data['intrinsics'][idx],  # Intrinsic camera parameters (shape: [samples, 3, 3])
            'keypoints_2d': self.data['keypoints_2d'][idx],  # 2D keypoints (shape: [samples, 24, 2])
            'GT_pose': self.data['GT_pose'][idx],        # Ground truth 3D rotation parameters (shape: [samples, 72])
            'GT_beta': self.data['GT_beta'][idx],        # Ground truth body shape parameters (shape: [samples, 10])
        }
        return sample

def mpjpe_pampjpe_avgpelvis(pred_vertices, J_regressor, target_vertices):
    J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
    pred_j3ds = torch.matmul(J_regressor_batch, pred_vertices)
    pred_j3ds = pred_j3ds[:, H36M_TO_J14, :]
    target_j3ds = torch.matmul(J_regressor_batch, target_vertices)
    target_j3ds = target_j3ds[:, H36M_TO_J14, :]

    pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
    target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0
    pred_j3ds -= pred_pelvis
    target_j3ds -= target_pelvis

    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
    mpjpe = np.mean(errors) * 1000
    pa_mpjpe = np.mean(errors_pa) * 1000
    mpvpe = compute_error_verts_torch(target_verts=target_vertices, pred_verts=pred_vertices).cpu().numpy()
    return mpjpe, pa_mpjpe, np.max(errors_pa*1000), errors, errors_pa, mpvpe


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='The evaluation code for 3D human mesh refinement named KITRO')
    parser.add_argument('--data_path', type=str, default='data/ProcessedData_CLIFFpred_w2DKP_3dpw.pt',
                        help='Path to the saved .pth file containing the SMPL estimates data')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for DataLoader')
    parser.add_argument('--n_refine_loop', default=10, type=int, help='Number of total KITRO loops')
    parser.add_argument('--shape_opti_n_iter', default=10, type=int, help='iteration number for each shape refinement step in one KITRO loop')

    args = parser.parse_args()
    
    kitro_config = {
        'shape_opti_n_iter': args.shape_opti_n_iter,
        'n_refine_loop': args.n_refine_loop,
    }

    dataset = SMPL_Estimates_Dataset(args.data_path)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # SMPL related
    J_regressor = torch.from_numpy(np.load(config.JOINT_REGRESSOR_H36M)).float()
    smpl = SMPL_59(config.SMPL_MODEL_DIR, create_transl=False).to(device)

    quant_mpjpe = {}
    quant_pampjpe = {}
    quant_mpvpe = {}
    pbar = tqdm(dataloader, desc='Processing')
    for batch in pbar:
        # batch should contain dictionaries with keys: 'pred_theta', 'pred_beta', 'pred_cam', 'intrinsics', 'keypoints_2d', 'GT_pose', 'GT_beta'
        curr_batch_size = batch['pred_theta'].shape[0]
        init_smpl_estimate = {
            'pred_theta': batch['pred_theta'].to(device),  # Predicted 3D rotation matrix (shape: [samples, 24, 3, 3])
            'pred_beta': batch['pred_beta'].to(device),    # Predicted body shape parameters (shape: [b, 10])
            'pred_cam': batch['pred_cam'].to(device),      # Predicted camera translation (shape: [b, 3])
        }
        evidence_2d = {
            'intrinsics': batch['intrinsics'].to(device),  # Intrinsic camera parameters (shape: [b, 3, 3])
            'keypoints_2d': batch['keypoints_2d'].to(device),  # 2D keypoints (shape: [b, 24, 2])
        }
        with torch.no_grad():
            updated_smpl_output = KITRO_refine(curr_batch_size, init_smpl_estimate=init_smpl_estimate, evidence_2d=evidence_2d, J_regressor=J_regressor, kitro_cfg=kitro_config, smpl=smpl)
            refined_thetas = updated_smpl_output['refined_thetas']
            refined_shape = updated_smpl_output['refined_shape']
            refined_cam = updated_smpl_output['refined_cam']
            updated_smpl = smpl(betas=refined_shape, body_pose=refined_thetas[:, 1:], global_orient=refined_thetas[:, 0].unsqueeze(1), pose2rot=False)
            updated_vertices = updated_smpl.vertices

            # Calculate Joint ERROR with ground truth
            gt_smpl = smpl(global_orient=batch['GT_pose'].to(device)[:,:3], body_pose=batch['GT_pose'].to(device)[:,3:],    betas=batch['GT_beta'].to(device))
            gt_vertices = gt_smpl.vertices
            _, _, _, error, r_error, mpvpe = mpjpe_pampjpe_avgpelvis(updated_vertices, J_regressor, gt_vertices)
        for ii, p in enumerate(batch['imgname'][:len(r_error)]):
            seqName = os.path.basename( os.path.dirname(p))
            if seqName not in quant_mpjpe.keys():
                quant_mpjpe[seqName] = []
                quant_pampjpe[seqName] = []
                quant_mpvpe[seqName] = []

            quant_mpjpe[seqName].append(error[ii]) 
            quant_pampjpe[seqName].append(r_error[ii])
            quant_mpvpe[seqName].append(mpvpe[ii])
        
        list_mpjpe = np.hstack([ quant_mpjpe[k] for k in quant_mpjpe])
        list_pampjpe = np.hstack([ quant_pampjpe[k] for k in quant_pampjpe])
        list_mpvpe = np.hstack([ quant_mpvpe[k] for k in quant_mpvpe])
        postfix = OrderedDict([('Seq', seqName),('MPJPE', "{:.02f} mm".format(np.mean(error) * 1000)),('PAMPJPE', "{:.02f} mm".format(np.mean(r_error) * 1000)),('MPVPE', "{:.02f} mm".format(np.mean(mpvpe) * 1000)),('Total_MPJPE', "{:.02f} mm".format(np.hstack(list_mpjpe).mean() * 1000)),('Total_PAMPJPE', "{:.02f} mm".format(np.hstack(list_pampjpe).mean() * 1000)),('Total_MPVPE', "{:.02f} mm".format(np.hstack(list_mpvpe).mean() * 1000))])
        pbar.set_postfix(ordered_dict=postfix, refresh=True)

    # Save logs
    list_mpjpe = np.hstack([ quant_mpjpe[k] for k in quant_mpjpe])
    list_pampjpe = np.hstack([ quant_pampjpe[k] for k in quant_pampjpe])
    list_mpvpe = np.hstack([ quant_mpvpe[k] for k in quant_mpvpe])

    output_str ='SeqNames; '
    for seq in quant_mpjpe:
        output_str += seq + ';'
    output_str +='\n MPJPE; '
    quant_mpjpe_avg_mm = np.hstack(list_mpjpe).mean()*1000
    output_str += "Avg {:.02f} mm; ".format( quant_mpjpe_avg_mm)
    for seq in quant_mpjpe:
        output_str += '{:.02f}; '.format(1000 * np.hstack(quant_mpjpe[seq]).mean())
    output_str +='\n PA-MPJPE; '
    quant_recon_error_avg_mm = np.hstack(list_pampjpe).mean()*1000
    output_str +="Avg {:.02f}mm; ".format( quant_recon_error_avg_mm )
    for seq in quant_pampjpe:
        output_str += '{:.02f}; '.format(1000 * np.hstack(quant_pampjpe[seq]).mean())
    output_str +='\n MPVPE; '
    quant_mpvpe_avg_mm = np.hstack(list_mpvpe).mean()*1000
    output_str +="Avg {:.02f}mm; ".format( quant_mpvpe_avg_mm )
    for seq in quant_mpvpe:
        output_str += '{:.02f}; '.format(1000 * np.hstack(quant_mpvpe[seq]).mean())
    print(output_str)
  
    data_to_save = {
        'MPJPE': {
            'avg_mm': quant_mpjpe_avg_mm,
            'values': {seq: 1000 * np.hstack(quant_mpjpe[seq]).mean() for seq in quant_mpjpe}
        },
        'PA_MPJPE': {
            'avg_mm': quant_recon_error_avg_mm,
            'values': {seq: 1000 * np.hstack(quant_pampjpe[seq]).mean() for seq in quant_pampjpe}
        },
        'MPVPE': {
            'avg_mm': quant_mpvpe_avg_mm,
            'values': {seq: 1000 * np.hstack(quant_mpvpe[seq]).mean() for seq in quant_mpvpe}
        }
    }
    # Use the current date in the filename
    current_date = datetime.now().strftime("%Y-%m-%d")
    json_filename = os.path.join("logs", f"output_log_{args.data_path.split('/')[-1]}_{current_date}.json")
    with open(json_filename, 'w') as json_file:
        json.dump(data_to_save, json_file, indent=4)
    print(f"The output has been saved to {json_filename}")