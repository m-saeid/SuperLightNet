import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_util import *



class Normalization(nn.Module):
    def __init__(self, d, mode="center", use_xyz=True, **kwargs):
        super(Normalization, self).__init__()
        self.use_xyz = use_xyz
        if mode.lower() == 'center':
            self.mode = 'center'
        elif mode.lower() == 'anchor':
            self.mode = 'anchor'
        else:
            print(f"Unrecognized Normalization.mode: {mode}! >>> None.")
            self.mode = None
        if self.mode is not None:
            add_d=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,d + add_d]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, d + add_d]))
        
    def forward(self, xyz_sampled, f_sampled, xyz_grouped, f_grouped):
        b,s,k,_ = xyz_grouped.shape
        # USE_XYZ
        f_grouped = torch.cat([f_grouped, xyz_grouped],dim=-1) if self.use_xyz else f_grouped
        # NORMALIZE
        if self.mode is not None:
            if self.mode =="center":   # False
                mean = torch.mean(f_grouped, dim=2, keepdim=True)
            if self.mode =="anchor":   # True
                mean = torch.cat([f_sampled, xyz_sampled],dim=-1) if self.use_xyz else f_sampled # True (2,512,19)=cat((2,512,16),(2,512,3))
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]    (2,512,1,19)
            std = torch.std((f_grouped-mean).reshape(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # [b,1,1,1]
            f_grouped = (f_grouped-mean)/(std + 1e-5) #  (2,512,24,19) = (2,512,24,19)/(2,1,1,1)  

            f = self.affine_alpha*f_grouped + self.affine_beta

        f_grouped = torch.cat([f_grouped, f_sampled.view(b, s, 1, -1).repeat(1, 1, k, 1)], dim=-1)
        return f_grouped       # 2,512,24,35
    