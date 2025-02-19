import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# PosE for Raw-point Embedding 
class TPE(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz): # B,C,N
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float()#.cuda()     
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed
    

tp = TPE(3,18,1000,100)
xyz = torch.rand(2,3,1024)
print(tp(xyz).shape)