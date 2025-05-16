import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *


from pointnet2_ops import pointnet2_utils
from pytorch3d.ops import sample_farthest_points

class Sampling(nn.Module):
    def __init__(self, mode='fps', s=512, **kwargs):
        super(Sampling, self).__init__()
        self.mode = mode
        self.s = s

    def forward(self, xyz, f):
        if self.mode == "fps":

            mode = 'pointops2'

            if mode =='pytorch3d':
                xyz_sampled, idx = sample_farthest_points(xyz, K=self.s)#.long()
                #idx = idx.long()
            elif mode == 'pointops2':
                idx = pointnet2_utils.furthest_point_sample(xyz, self.s).long()  # [B, npoint]
                xyz_sampled = index_points(xyz, idx)
            elif mode == 'pytorch':
                idx = farthest_point_sample(xyz, self.s).long()
                xyz_sampled = index_points(xyz, idx)


            f_sampled = index_points(f, idx) if f is not None else None
            return xyz_sampled, f_sampled