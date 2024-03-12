import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from lib.utils.geometry import *

J24_INDEX_IN_59 = [8,12,9,49,13,10,50,30,25,51,52,53,1,54,55,56,5,2,6,3,7,4,57,58]
PARENTS = [-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,21]

def perspective_projection(points, rotation, trans, K):
    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + trans.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:, :, -1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


def batch_cal_swing(t, p): 
    # The function calculates swing rotation matrix that can rotate bone from direction t to direction p. Details refer to Sec.4.2 in our main paper.

    batch_size = t.shape[0]
    t = t.unsqueeze(-1) # [b, 3, 1]
    p = p.unsqueeze(-1) # [b, 3, 1]
    t_norm = torch.norm(t, dim=1, keepdim=True) # [b, 1, 1]
    p_norm = torch.norm(p, dim=1, keepdim=True) # [b, 1, 1]
    
    # rot axis
    axis = torch.cross(t, p, dim=1) # [b, 3, 1]
    axis_norm = torch.norm(axis, dim=1, keepdim=True) # [b, 1, 1]
    axis = axis / (axis_norm + 1e-8)  # [b, 3, 1]

    cos = torch.sum(t * p, dim=1, keepdim=True) / (t_norm * p_norm + 1e-8) # [b, 1]
    sin = axis_norm / (t_norm * p_norm + 1e-8)

    # Convert location revolve to rot_mat by rodrigues
    rx, ry, rz = torch.split(axis, 1, dim=1)
    zeros = torch.zeros((batch_size, 1, 1)).cuda()

    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))
    ident = torch.eye(3).unsqueeze(dim=0).cuda()
    rot_mat_loc = ident + sin * K + (1 - cos) * torch.matmul(K, K)
    return rot_mat_loc

def batch_cal_2solutions(Ap, Bp, OA_len, bone_len, inv_intrinsics):
    # The function calculates the two possible solutions for bone direction based on the 2DKP, parent joint depth, bone length, and camera intrinsic. Details refer to Figure 3 in our main paper.

    # uv space to camera space
    OAp = torch.einsum('bij,bj->bi', inv_intrinsics, Ap)
    OBp = torch.einsum('bij,bj->bi', inv_intrinsics, Bp)

    # print(OAp, OBp)
    OAp_norm = torch.norm(OAp, dim=-1, keepdim=True) 
    OBp_norm = torch.norm(OBp, dim=-1, keepdim=True)
    a = OAp / OAp_norm 
    b = OBp / OBp_norm
    cos = torch.sum(OAp * OBp, dim=-1, keepdim=True) / (OAp_norm * OBp_norm) 
    sin = torch.norm(torch.cross(OAp, OBp, dim=-1), dim=-1, keepdim=True) / (OAp_norm * OBp_norm)
    AF = OA_len * (cos * b - a)

    # Depending on the accuracy of the estimated terms, the square root term may become negative. For numerical stability, we rectify it to 0.
    inner = bone_len**2 - OA_len**2 * sin**2
    scale = torch.sqrt(torch.where(inner > 0, inner, torch.tensor(0).float().cuda())) 

    FB = scale * b  
    AB1 = AF + FB  
    AB2 = AF - FB  

    #OB lenth (child joint depth)
    OB_len1 = OA_len * cos + scale
    OB_len2 = OA_len * cos - scale

    return torch.cat((AB1.unsqueeze(1), AB2.unsqueeze(1)), dim=1), torch.cat((OB_len1.unsqueeze(1), OB_len2.unsqueeze(1)), dim=1)


def batch_get_pelvis_orient_svd(t0,t1,t2,p0,p1,p2):
    # adopted from HybrIK(link https://github.com/Jeff-sjtu/HybrIK/blob/9b8681dcf3c902dd5dacc01520ba04982990e1e2/hybrik/models/layers/smpl/lbs.py#L937)

    rest_mat = torch.cat([t0[:, :, None], t1[:, :, None], t2[:, :, None]], dim=2).clone()
    target_mat = torch.cat([p0[:, :, None], p1[:, :, None], p2[:, :, None]], dim=2).clone()

    S = rest_mat.bmm(target_mat.transpose(1, 2))

    mask_zero = S.sum(dim=(1, 2))

    S_non_zero = S[mask_zero != 0].reshape(-1, 3, 3)

    # U, _, V = torch.svd(S_non_zero)
    device = S_non_zero.device
    U, _, V = torch.svd(S_non_zero.cpu())
    U = U.to(device=device)
    V = V.to(device=device)

    rot_mat = torch.zeros_like(S)
    rot_mat[mask_zero == 0] = torch.eye(3, device=S.device)

    # rot_mat_non_zero = torch.bmm(V, U.transpose(1, 2))
    det_u_v = torch.det(torch.bmm(V, U.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(U.shape[0], -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v
    rot_mat_non_zero = torch.bmm(torch.bmm(V, det_modify_mat), U.transpose(1, 2))

    rot_mat[mask_zero != 0] = rot_mat_non_zero

    assert torch.sum(torch.isnan(rot_mat)) == 0, ('rot_mat', rot_mat)

    return rot_mat


def batch_cal_2d_error_after_align(refine_output_i_joints, chain_index, cam_R, cam_t, intrinsics, keypoints_2d, alignjid):
    # Calculate 2D error after alignment on root joint

    myj2d = perspective_projection(refine_output_i_joints, cam_R, cam_t, intrinsics)
    myj2d = torch.cat((myj2d, torch.ones(myj2d.shape[0], myj2d.shape[1], 1).type_as(myj2d)), dim=-1).squeeze(0).float().cuda()[:,J24_INDEX_IN_59]
    delta = myj2d[:,alignjid:alignjid+1] - keypoints_2d[:,alignjid:alignjid+1]
    j2d_delta = keypoints_2d + delta
    return torch.mean(torch.index_select((myj2d - j2d_delta)**2, 1, chain_index), dim=(-1,-2))


def optimize_shape(pred_shape, refined_thetas, refined_cam, intrinsics, keypoints_2d, smpl, num_iterations = 50):
    ############## Optimize Shape ###############
    batch_size = pred_shape.shape[0]
    refine_output_curr = smpl(
            betas=pred_shape,
            body_pose=refined_thetas[:, 1:],
            global_orient=refined_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
    )

    # Align 2D based on parent joint
    myj2d = perspective_projection(refine_output_curr.joints, torch.eye(3).repeat(batch_size,1,1).cuda(), refined_cam, intrinsics)
    myj2d = torch.cat((myj2d, torch.ones(myj2d.shape[0], myj2d.shape[1], 1).type_as(myj2d)), dim=-1).float().cuda()
    myj2d = myj2d[:,J24_INDEX_IN_59]
    delta = myj2d[:,0:1] - keypoints_2d[:,0:1]
    keypoints_2d_delta = keypoints_2d + delta

    bonelen_pred = torch.stack([torch.norm(refine_output_curr.joints[:,J24_INDEX_IN_59[child]] - refine_output_curr.joints[:,J24_INDEX_IN_59[PARENTS[child]]], dim=-1, keepdim=True) for child in range(1,24)], dim=0)
    proj_bonelen_pred = torch.stack([torch.norm(myj2d[:,child] - myj2d[:,PARENTS[child]], dim=-1, keepdim=True) for child in range(1,24)], dim=0) 
    proj_bonelen_gt = torch.stack([torch.norm(keypoints_2d_delta[:,child] - keypoints_2d_delta[:,PARENTS[child]], dim=-1, keepdim=True) for child in range(1,24)], dim=0)
    expect_bonelen = (proj_bonelen_gt/proj_bonelen_pred) * bonelen_pred
    with torch.enable_grad():
        # Initialize the shape parameters
        shape_params = pred_shape.detach().clone().requires_grad_(True)
        refined_thetas.requires_grad_(False)
        rotation_matrix = torch.eye(3).repeat(batch_size, 1, 1).cuda()
        rotation_matrix.requires_grad_(False)
        refined_cam.requires_grad_(False)
        intrinsics.requires_grad_(False)
        keypoints_2d_delta.requires_grad_(False)
        proj_bonelen_gt.requires_grad_(False)
        expect_bonelen.requires_grad_(False)
        # Set up the optimizer
        optimizer = optim.Adam([shape_params], lr=0.1)
        loss = nn.L1Loss()
        for opt_step in range(num_iterations):
            optimizer.zero_grad()
            keypoints_3d = smpl(
                betas=shape_params,
                body_pose=refined_thetas[:, 1:],
                global_orient=refined_thetas[:, 0].unsqueeze(1),
                pose2rot=False,
            ).joints
            projected_keypoints = perspective_projection(keypoints_3d, rotation_matrix, refined_cam, intrinsics) # Shape (batch_size, num_keypoints, 2)
            proj_bonelen_pred = torch.stack([torch.norm(projected_keypoints[:,J24_INDEX_IN_59[child]] - projected_keypoints[:,J24_INDEX_IN_59[PARENTS[child]]], dim=-1, keepdim=True) for child in range(1,24)], dim=0)
            bonelen_error = loss(proj_bonelen_pred, proj_bonelen_gt) 
            error = bonelen_error
            error.backward()
            # Update the shape parameters
            optimizer.step()
    return shape_params


def estimate_cam_location(refine_output_curr, keypoints_2d, intrinsics):
    ############## Estimate Camera ###############
    batch_size = refine_output_curr.joints.shape[0]
    pred_cam = []
    for i in range(batch_size):
        pred_cam_t = estimate_cam_translation(refine_output_curr.joints[i:i+1,J24_INDEX_IN_59], keypoints_2d[i:i+1], focal_length=[intrinsics[i,0,0].cpu().item(), intrinsics[i,1,1].cpu().item()], img_size=[intrinsics[i,0,2].cpu().item()*2, intrinsics[i,1,2].cpu().item()*2])
        pred_cam.append(pred_cam_t)
    return torch.stack(pred_cam, dim=0).squeeze(1)

def gram_schmidt_batch(A):
    batch_size, n, _ = A.shape
    Q = torch.zeros_like(A)
    for k in range(n):
        qq = A[:, :, k]
        for i in range(k):
            qq = qq - torch.sum(A[:, :, k] * Q[:, :, i], dim=-1, keepdim=True) * Q[:, :, i]
        # Normalize qq
        qq = qq / torch.norm(qq, dim=1, keepdim=True)
        Q[:, :, k] = qq
    return Q

def solution_calculation_chain(refine_output_curr, inv_intrinsics, keypoints_2d, reproj2d, reference_joints, Oparent_len, chain):
    # hypothesis calculation
    batch_size = refine_output_curr.joints.shape[0]
    n_branch = 1
    Oparent_len = Oparent_len[:,None,:]
    result = {}
    similarity = torch.ones(batch_size)[:,None].cuda()
    rotmat = torch.eye(3)[None, None, :].repeat(batch_size,1,1,1).cuda()
    for child in chain:
        parent = PARENTS[child]
        child_id = J24_INDEX_IN_59[child]
        parent_id = J24_INDEX_IN_59[parent]

        # Align 2D based on parent joint
        delta = reproj2d[:,parent:parent+1] - keypoints_2d[:,parent:parent+1]
        keypoints_2d_delta = keypoints_2d + delta
        bonelen = torch.norm(refine_output_curr.joints[:,child_id] - refine_output_curr.joints[:,parent_id], dim=-1, keepdim=True)
        two_solu, two_len = batch_cal_2solutions(keypoints_2d_delta[:, parent][:,None,:].repeat(1,n_branch,1).reshape(batch_size*n_branch, 3),\
                                                 keypoints_2d_delta[:, child][:,None,:].repeat(1,n_branch,1).reshape(batch_size*n_branch, 3),\
                                                 Oparent_len.reshape(batch_size*n_branch, 1),\
                                                 bonelen[:,None,:].repeat(1,n_branch,1).reshape(batch_size*n_branch, 1),\
                                                 inv_intrinsics[:,None,:,:].repeat(1,n_branch,1,1).reshape(batch_size*n_branch, 3,3))

        # caluculate similarity
        result[child] = two_solu.reshape(batch_size, n_branch*2, 3)
        reference = torch.einsum('brij,bj->bri', rotmat, (reference_joints[:,child_id] - reference_joints[:,parent_id]))[:,:,None,:].repeat(1,1,2,1).reshape(batch_size, n_branch*2, 3)
        similarity = similarity[:,:,None].repeat(1,1,2).reshape(batch_size, n_branch*2) * ((result[child] * reference).sum(-1) / (torch.norm(result[child], dim=-1) * torch.norm(reference, dim=-1)) + 1) /2

        # update for next joint
        Oparent_len = two_len.reshape(batch_size,n_branch*2,1)
        n_branch*=2
        R_swing = batch_cal_swing(reference.reshape(batch_size*n_branch,3), two_solu.reshape(batch_size*n_branch,3)).reshape(batch_size,n_branch,3,3)
        rotmat = torch.einsum('brij,brjk->brik', R_swing, rotmat[:,:,None,:,:].repeat(1,1,2,1,1).reshape(batch_size,n_branch,3,3))

    return similarity


def solution_selection_chain(similarity, chain, selection):
    # hypothesis selection based on original HMR prediction. 
    batch_size = selection.shape[0]
    choise = torch.argmax(similarity, dim=-1)
    for i in reversed(chain):
        selection[:, i] = choise % 2
        choise = torch.div(choise, 2, rounding_mode='floor')
        similarity = torch.max(similarity.reshape(batch_size,-1,2), dim=-1)[0]
    return selection

def solution_selection(refined_thetas, refined_shape, smpl, refined_cam, intrinsics, inv_intrinsics, keypoints_2d, reference_joints):
    # Decision tree formulation and hypothesis selection
    batch_size = refined_thetas.shape[0]
    refine_output_curr = smpl(
            betas=refined_shape,
            body_pose=refined_thetas[:, 1:],
            global_orient=refined_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
    )
    Oparent_len = torch.sign((refine_output_curr.joints + refined_cam.unsqueeze(1))[:,:,-1:]) * torch.norm(refine_output_curr.joints + refined_cam.unsqueeze(1), dim=-1, keepdim=True)
    reproj2d = perspective_projection(refine_output_curr.joints, torch.eye(3).repeat(batch_size,1,1).cuda(), refined_cam, intrinsics)
    reproj2d = torch.cat((reproj2d, torch.ones(reproj2d.shape[0], reproj2d.shape[1], 1).type_as(reproj2d)), dim=-1).float().cuda()[:,J24_INDEX_IN_59]

    selection = -torch.ones([batch_size,24]).long().cuda()

    # left leg
    chain = [1,4,7,10]
    similarity_ll = solution_calculation_chain(refine_output_curr, inv_intrinsics, keypoints_2d, reproj2d, reference_joints, Oparent_len = Oparent_len[:, 8], chain=chain)
    selection = solution_selection_chain(similarity_ll, chain, selection)

    # right leg
    chain = [2,5,8,11]
    similarity_rl = solution_calculation_chain(refine_output_curr, inv_intrinsics, keypoints_2d, reproj2d, reference_joints, Oparent_len = Oparent_len[:, 8], chain=chain)
    selection = solution_selection_chain(similarity_rl, chain, selection)

    # left arm
    chain_la = [3,6,9,13,16,18,20,22]
    similarity_la = solution_calculation_chain(refine_output_curr, inv_intrinsics, keypoints_2d, reproj2d, reference_joints, Oparent_len = Oparent_len[:, 8], chain=chain_la)

    # right arm
    chain_ra = [3,6,9,14,17,19,21,23]
    similarity_ra = solution_calculation_chain(refine_output_curr, inv_intrinsics, keypoints_2d, reproj2d, reference_joints, Oparent_len = Oparent_len[:, 8], chain=chain_ra)

    # head
    chain_h = [3,6,9,12,15]
    similarity_h = solution_calculation_chain(refine_output_curr, inv_intrinsics, keypoints_2d, reproj2d, reference_joints, Oparent_len = Oparent_len[:, 8], chain=chain_h)

    # Since 3 chains (left arm, right arm, head) share [3,6,9], considering all three chains for their solutions selection 
    chain = [3,6,9]
    similarity_369 = similarity_la.reshape(batch_size,8,-1).max(-1)[0] *\
                     similarity_ra.reshape(batch_size,8,-1).max(-1)[0] *\
                     similarity_h.reshape(batch_size,8,-1).max(-1)[0]
    selection = solution_selection_chain(similarity_369, chain, selection)

    chain = [13,16,18,20,22]
    selection = solution_selection_chain(similarity_la.reshape(batch_size,2,2,2,-1)[torch.arange(batch_size),selection[:,3].long(),selection[:,6].long(),selection[:,9].long()],\
                                                     chain, selection)
    
    chain = [14,17,19,21,23]
    selection = solution_selection_chain(similarity_ra.reshape(batch_size,2,2,2,-1)[torch.arange(batch_size),selection[:,3].long(),selection[:,6].long(),selection[:,9].long()],\
                                                     chain, selection)
    
    chain = [12,15]
    selection = solution_selection_chain(similarity_h.reshape(batch_size,2,2,2,-1)[torch.arange(batch_size),selection[:,3].long(),selection[:,6].long(),selection[:,9].long()],\
                                                     chain, selection)


    return selection

def refine_alignKT(refined_thetas, refined_shape, smpl, refined_cam, intrinsics, inv_intrinsics, keypoints_2d, selection, origin_thetas):
    # refine pose parameters along kinematic-tree based on the chosen hypothesis
    batch_size = refined_thetas.shape[0]
    refine_output_curr = smpl(
            betas=refined_shape,
            body_pose=refined_thetas[:, 1:],
            global_orient=refined_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
    )
    refine_output_origin = smpl(
            betas=refined_shape,
            body_pose=origin_thetas[:, 1:],
            global_orient=origin_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
    )

    already_refined_index = []
    for i in range(24):
        refined_thetas_copy = refined_thetas.clone()
        parent_id = J24_INDEX_IN_59[i]
        child = np.argwhere(np.array(PARENTS) == i).squeeze(-1)

        # parent joint depth from camera O
        Oparent_len = torch.sign((refine_output_curr.joints + refined_cam.unsqueeze(1))[:,parent_id,-1:]) * torch.norm(refine_output_curr.joints[:, parent_id] + refined_cam, dim=-1, keepdim=True)

        # Align 2D based on parent joint
        myj2d = perspective_projection(refine_output_curr.joints, torch.eye(3).repeat(batch_size,1,1).cuda(), refined_cam, intrinsics)
        myj2d = torch.cat((myj2d, torch.ones(myj2d.shape[0], myj2d.shape[1], 1).type_as(myj2d)), dim=-1).float().cuda()[:,J24_INDEX_IN_59]
        delta = myj2d[:,i:i+1] - keypoints_2d[:,i:i+1]
        keypoints_2d_delta = keypoints_2d + delta

        if i == 0 or i == 9: # joint 0 and 9 has 3 child joints.
            # child0
            child_id0 = J24_INDEX_IN_59[child[0]]
            two_solu0, two_len0 = batch_cal_2solutions(keypoints_2d_delta[:, i], keypoints_2d_delta[:, child[0]], Oparent_len, torch.norm(refine_output_curr.joints[:,child_id0] - refine_output_curr.joints[:,parent_id], dim=-1, keepdim=True), inv_intrinsics)
            # child1
            child_id1 = J24_INDEX_IN_59[child[1]]
            two_solu1, two_len1 = batch_cal_2solutions(keypoints_2d_delta[:, i], keypoints_2d_delta[:, child[1]], Oparent_len, torch.norm(refine_output_curr.joints[:,child_id1] - refine_output_curr.joints[:,parent_id], dim=-1, keepdim=True), inv_intrinsics)
            # child2
            child_id2 = J24_INDEX_IN_59[child[2]]
            two_solu2, two_len2 = batch_cal_2solutions(keypoints_2d_delta[:, i], keypoints_2d_delta[:, child[2]], Oparent_len, torch.norm(refine_output_curr.joints[:,child_id2] - refine_output_curr.joints[:,parent_id], dim=-1, keepdim=True), inv_intrinsics)

            # choose the solution given by the optimal path of the decision tree
            index0 = selection[:,child[0]]
            index1 = selection[:,child[1]]
            index2 = selection[:,child[2]]

            two_solu0_dir = two_solu0 / torch.norm(two_solu0, dim=-1, keepdim=True)
            two_solu1_dir = two_solu1 / torch.norm(two_solu1, dim=-1, keepdim=True)
            two_solu2_dir = two_solu2 / torch.norm(two_solu2, dim=-1, keepdim=True)
            pred_dir0 = refine_output_origin.joints[:,child_id0] - refine_output_origin.joints[:,parent_id]
            pred_dir0 = pred_dir0 / torch.norm(pred_dir0, dim=-1, keepdim=True)
            dir_similarity0 = torch.sum(pred_dir0.unsqueeze(1) * two_solu0_dir, dim=-1)
            pred_dir1 = refine_output_origin.joints[:,child_id1] - refine_output_origin.joints[:,parent_id]
            pred_dir1 = pred_dir1 / torch.norm(pred_dir1, dim=-1, keepdim=True)
            dir_similarity1 = torch.sum(pred_dir1.unsqueeze(1) * two_solu1_dir, dim=-1)
            pred_dir2 = refine_output_origin.joints[:,child_id2] - refine_output_origin.joints[:,parent_id]
            pred_dir2 = pred_dir2 / torch.norm(pred_dir2, dim=-1, keepdim=True)
            dir_similarity2 = torch.sum(pred_dir2.unsqueeze(1) * two_solu2_dir, dim=-1)

            # use the similarity as the refinement weight
            refineweight0 = torch.softmax(dir_similarity0, dim=-1)[torch.arange(batch_size), index0.squeeze(-1).long()]
            refineweight1 = torch.softmax(dir_similarity1, dim=-1)[torch.arange(batch_size), index1.squeeze(-1).long()]
            refineweight2 = torch.softmax(dir_similarity2, dim=-1)[torch.arange(batch_size), index2.squeeze(-1).long()]

            # calculate the swing rotation matrix
            R_swing = batch_get_pelvis_orient_svd(refine_output_curr.joints[:,child_id0] - refine_output_curr.joints[:,parent_id], refine_output_curr.joints[:,child_id1] - refine_output_curr.joints[:,parent_id], refine_output_curr.joints[:,child_id2] - refine_output_curr.joints[:,parent_id],\
                                refineweight0[:,None] * two_solu0[torch.arange(batch_size), index0.squeeze(-1).long()] + (1.0-refineweight0)[:,None] * (refine_output_curr.joints[:,child_id0] - refine_output_curr.joints[:,parent_id]),
                                refineweight1[:,None] * two_solu1[torch.arange(batch_size), index1.squeeze(-1).long()] + (1.0-refineweight1)[:,None] * (refine_output_curr.joints[:,child_id1] - refine_output_curr.joints[:,parent_id]),
                                refineweight2[:,None] * two_solu2[torch.arange(batch_size), index2.squeeze(-1).long()] + (1.0-refineweight2)[:,None] * (refine_output_curr.joints[:,child_id2] - refine_output_curr.joints[:,parent_id])
                                )

            # pose parameter update
            if i==0:
                R_parent = torch.eye(3).repeat(batch_size,1,1).cuda()
            else:
                j = PARENTS[i]
                R_parent = refined_thetas_copy[:,j]
                while j != 0:
                    j = PARENTS[j]
                    R_parent = torch.bmm(refined_thetas_copy[:,j], R_parent)
            refined_thetas[:,i] = torch.bmm(torch.bmm(torch.bmm(torch.transpose(R_parent, -1, -2), R_swing), R_parent), refined_thetas_copy[:,i]) # Equation (19) in main paper
            refined_thetas[:,i] = gram_schmidt_batch(refined_thetas[:,i]) # just prevent from error accumulating, keep the matrix as rotation matrix
            already_refined_index += list(child)
        else: # other joints only have one child
            if len(child) == 0: # leaf node has no child
                continue
            child_id = J24_INDEX_IN_59[child[0]]
            two_solu, two_len = batch_cal_2solutions(keypoints_2d_delta[:, i], keypoints_2d_delta[:, child[0]], Oparent_len, torch.norm(refine_output_curr.joints[:,child_id] - refine_output_curr.joints[:,parent_id], dim=-1, keepdim=True), inv_intrinsics)

            # choose the solution given by the optimal path of the decision tree
            index = selection[:,child[0]]
            choice = two_solu[torch.arange(batch_size), index.squeeze(-1).long()]
            
            two_solu_dir = two_solu / torch.norm(two_solu, dim=-1, keepdim=True)
            pred_dir = refine_output_origin.joints[:,child_id] - refine_output_origin.joints[:,parent_id]
            pred_dir = pred_dir / torch.norm(pred_dir, dim=-1, keepdim=True)
            dir_similarity = torch.sum(pred_dir.unsqueeze(1) * two_solu_dir, dim=-1)

            # use the similarity as the refinement weight
            refineweight = torch.softmax(dir_similarity, dim=-1)[torch.arange(batch_size), index.squeeze(-1).long()]

            # calculate the swing rotation matrix
            R_swing = batch_cal_swing(refine_output_curr.joints[:,child_id] - refine_output_curr.joints[:,parent_id], 
                                      refineweight[:,None] * choice + (1.0-refineweight)[:,None] * (refine_output_curr.joints[:,child_id] - refine_output_curr.joints[:,parent_id]))

            # pose parameter update for parent joint
            j = PARENTS[i]
            R_parent = refined_thetas_copy[:,j]
            while j != 0:
                j = PARENTS[j]
                R_parent = torch.bmm(refined_thetas_copy[:,j], R_parent)
            refined_thetas[:,i] = torch.bmm(torch.bmm(torch.bmm(torch.transpose(R_parent, -1, -2), R_swing), R_parent), refined_thetas_copy[:,i]) # Equation (19) in main paper
            refined_thetas[:,i] = gram_schmidt_batch(refined_thetas[:,i]) # just prevent from error accumulating, keep the matrix as rotation matrix
            already_refined_index += list(child)

        # rewind theta if the 2D reprojection gets even worser after refinemnet
        old_2d_err = batch_cal_2d_error_after_align(refine_output_curr.joints, torch.tensor([ii for ii in already_refined_index]).cuda(), torch.eye(3).repeat(batch_size,1,1).cuda(), refined_cam, intrinsics, keypoints_2d, alignjid=i)
        refine_output_curr = smpl(
            betas=refined_shape,
            body_pose=refined_thetas[:, 1:],
            global_orient=refined_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
        )
        new_2d_err = batch_cal_2d_error_after_align(refine_output_curr.joints, torch.tensor([ii for ii in already_refined_index]).cuda(), torch.eye(3).repeat(batch_size,1,1).cuda(), refined_cam, intrinsics, keypoints_2d, alignjid=i)
        indicator = new_2d_err < old_2d_err
        refined_thetas = torch.where(indicator[:,None,None,None].repeat(1,24,3,3) == True, refined_thetas, refined_thetas_copy)
        refine_output_curr = smpl(
            betas=refined_shape,
            body_pose=refined_thetas[:, 1:],
            global_orient=refined_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
        )
        # update original pose parameters (for refinement weight)
        tmp_thetas = refined_thetas.clone()
        tmp_thetas[:, i+1:] = origin_thetas[:, i+1:]
        refine_output_origin = smpl(
            betas=refined_shape,
            body_pose=tmp_thetas[:, 1:],
            global_orient=tmp_thetas[:, 0].unsqueeze(1),
            pose2rot=False,
        )
    return refined_thetas


def KITRO_refine(batch_size, init_smpl_estimate, evidence_2d, J_regressor=None, kitro_cfg=None, smpl=None):
    shape_opti_n_iter = kitro_cfg['shape_opti_n_iter']
    n_refine_loop = kitro_cfg['n_refine_loop']

    ############## Initial Prediction ###############
    pred_thetas = init_smpl_estimate['pred_theta'] # [b,24,3,3]
    pred_beta = init_smpl_estimate['pred_beta'] # [b,10]
    pred_cam = init_smpl_estimate['pred_cam'] # [b,3]

    ############## 2D Evidence ###############
    keypoints_2d = evidence_2d['keypoints_2d']#[:,J24_INDEX_IN_59] # [b,24,2]
    keypoints_2d = torch.cat((keypoints_2d, torch.ones(keypoints_2d.shape[0], keypoints_2d.shape[1], 1).type_as(keypoints_2d)), dim=-1).float().cuda() # [b,24,3]
    intrinsics = evidence_2d['intrinsics']
    inv_intrinsics = torch.inverse(intrinsics)

    ############## KITRO Refinement ###############
    refined_thetas = pred_thetas.clone()
    refined_shape = pred_beta.clone()
    refined_cam = pred_cam
    for loop in range(n_refine_loop):
        refined_smpl = smpl(betas=refined_shape, body_pose=refined_thetas[:, 1:], global_orient=refined_thetas[:, 0].unsqueeze(1), pose2rot=False)
        
        ############## Estimate Camera ###############
        refined_cam = (refined_cam + estimate_cam_location(refined_smpl, keypoints_2d, intrinsics)) / 2

        ############## Optimize Shape ###############
        refined_shape = optimize_shape(refined_shape, refined_thetas, refined_cam, intrinsics, keypoints_2d, smpl, num_iterations = shape_opti_n_iter)

        ############## Refine Pose ###############
        # Decision tree formulation and hypothesis selection
        selection = solution_selection(refined_thetas, refined_shape, smpl, refined_cam, intrinsics, inv_intrinsics, keypoints_2d, reference_joints=smpl(betas=refined_shape, body_pose=pred_thetas[:, 1:], global_orient=pred_thetas[:, 0].unsqueeze(1), pose2rot=False).joints)
        # update theta based on selected hypothesis
        refined_thetas = refine_alignKT(refined_thetas, refined_shape, smpl, refined_cam, intrinsics, inv_intrinsics, keypoints_2d, selection, origin_thetas=pred_thetas.clone())

    updated_smpl_output = {
        'refined_thetas': refined_thetas,
        'refined_shape': refined_shape,
        'refined_cam': refined_cam
    }
    return updated_smpl_output

    
