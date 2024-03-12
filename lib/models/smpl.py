# Copyright (c) Facebook, Inc. and its affiliates.

#Modified from https://github.com/nkolot/SPIN/blob/master/LICENSE

import torch
import numpy as np
from smplx import SMPL as _SMPL
# from smplx.body_models import ModelOutput as SMPLOutput      #old version of smplx: 0.1.13
from smplx.body_models import SMPLOutput       #new version of smplx: 0.1.28
from smplx.lbs import vertices2joints

import lib.cores.config as config
import lib.cores.constants as constants
import lib.cores.jointorders as jointorders

import os.path as osp
EXTRA_DATA_DIR = 'data/extradata'
# Map joints to SMPL joints
JOINT_MAP = {
    'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
    'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
    'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
    'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
    'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
    'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
    'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
    'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
    'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
    'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
    'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
    'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
    'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
    'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
    'Spine (H36M)': 51, 'Jaw (H36M)': 52,
    'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
    'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27,
    'Spine': 3, 'Spine2': 6, 'Spine3': 9, 'left_foot': 10, 'right_foot': 11, 'left_collar': 13, 'right_collar': 14, 'head': 15, 'left_hand': 22, 'right_hand': 23 # these joints are missing in the previous 49 joints, we add them manually to form SMPL_59
}
JOINT_NAMES = [
    'OP Nose', 'OP Neck', 'OP RShoulder',
    'OP RElbow', 'OP RWrist', 'OP LShoulder',
    'OP LElbow', 'OP LWrist', 'OP MidHip',
    'OP RHip', 'OP RKnee', 'OP RAnkle',
    'OP LHip', 'OP LKnee', 'OP LAnkle',
    'OP REye', 'OP LEye', 'OP REar',
    'OP LEar', 'OP LBigToe', 'OP LSmallToe',
    'OP LHeel', 'OP RBigToe', 'OP RSmallToe', 'OP RHeel',
    'Right Ankle', 'Right Knee', 'Right Hip',
    'Left Hip', 'Left Knee', 'Left Ankle',
    'Right Wrist', 'Right Elbow', 'Right Shoulder',
    'Left Shoulder', 'Left Elbow', 'Left Wrist',
    'Neck (LSP)', 'Top of Head (LSP)',
    'Pelvis (MPII)', 'Thorax (MPII)',
    'Spine (H36M)', 'Jaw (H36M)',
    'Head (H36M)', 'Nose', 'Left Eye',
    'Right Eye', 'Left Ear', 'Right Ear',
    'Spine', 'Spine2', 'Spine3', 'left_foot', 'right_foot', 'left_collar', 'right_collar', 'head', 'left_hand', 'right_hand' # these joints are missing in the previous 49 joints, we add them manually to form SMPL_59
]

JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}
JOINT_REGRESSOR_TRAIN_EXTRA = osp.join(EXTRA_DATA_DIR, 'spin', 'J_regressor_extra.npy')
SMPL_MEAN_PARAMS = osp.join(EXTRA_DATA_DIR, 'spin', 'smpl_mean_params.npz')
SMPL_MODEL_DIR = osp.join(EXTRA_DATA_DIR, 'smpl')
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
SPIN49_TO_J14 = [30, 13, 12, 19, 10, 25, 7, 6, 5, 2, 3, 4, 1, 56]


class SMPL_19(_SMPL):   

    def __init__(self, *args, **kwargs):
        super(SMPL_19, self).__init__(*args, **kwargs)
        self.joint_map_smpl45_to_openpose19 = torch.tensor(jointorders.JOINT_MAP_SMPL45_TO_OPENPOSE18, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL_19, self).forward(*args, **kwargs)
        reordered_joints = smpl_output.joints[:, self.joint_map_smpl45_to_openpose19, :]       #Reordering

        new_output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=reordered_joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return new_output

class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)        #Additional 9 joints #Check doc/J_regressor_extra.png
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)               #[N, 24 + 21, 3]  + [N, 9, 3]
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output
    
class SMPL_59(_SMPL):  # We add the missing joints on kinematic-tree manually to form SMPL_59
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL_59, self).__init__(*args, **kwargs)
        joints = [JOINT_MAP[i] for i in JOINT_NAMES]
        J_regressor_extra = np.load(JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)


    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL_59, self).forward(*args, **kwargs)
        # import ipdb; ipdb.set_trace()
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output

def get_smpl_faces():
    print("Get SMPL faces")
    smpl = SMPL(SMPL_MODEL_DIR, batch_size=1, create_transl=False)
    return smpl.faces