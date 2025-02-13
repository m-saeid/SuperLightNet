import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder_util import *


class GPE(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(GPE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma

        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * self.in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()

        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)

    def forward(self, xyz):     # 128,1024,3
        # xyz = [B, N, 3] or [B, S, K, 3]

        if xyz.shape[1] == 3:
            xyz = xyz.permute(0,2,1)

        if self.out_dim == 0:
            return xyz

        if xyz.dim() not in {3, 4}:     #3
            raise ValueError("Input must be either [B, in_dim, N] or [B, in_dim, S, K]")

        embeds = []
        # Compute the RBF features for each channel in a loop  /// feat_val: [-0.3333, 0.3333] dayere / sigma: 0.3
        for i in range(self.in_dim):        # 3
            tmp = xyz[..., i : i + 1] - self.feat_val.to(xyz.device)    # [128,1024,2] = [128,1024,1] - [1,2]
            embed = -0.5 * tmp**2 / (self.sigma**2)                     # [128,1024,2] = [128,1024,2] / 0.3
            embeds.append(embed.exp())                                  # [128,1024,2]

        # Concatenate along the last dimension to get all features together
        position_embed = torch.cat(embeds, dim=-1)  # [B, ..., feat_num]    # [128,1024,6]

        # Select the required output dimensions using out_idx
        position_embed = torch.index_select(
            position_embed, -1, self.out_idx.to(xyz.device)
        )
        # [B, ..., out_dim]     [128,1024,6]

        # # Reshape based on the original input dimensions
        # if xyz.dim() == 3:
        #     b, _, n = xyz.shape
        #     position_embed = position_embed.permute(0, 2, 1).reshape(b, self.out_dim, n)
        #     # [B, out_dim, N]
        # elif xyz.dim() == 4:
        #     b, _, s, k = xyz.shape
        #     position_embed = position_embed.permute(0, 3, 1, 2).reshape(
        #         b, self.out_dim, s, k
        #     )
        #     # [B, feat_num, S, K]

        return position_embed.permute(0,2,1)  # [B, ..., out_dim]  [128,1024,6]


class Embedding(nn.Module):
    def __init__(self, in_ch, out_ch, mode, sigma=0.3):
        super(Embedding, self).__init__()
        # self.in_ch = in_ch
        # self.out_ch = out_ch
        # self.mode = mode
        if mode == 'mlp':
            self.embd = mlp(in_ch, out_ch)
        elif mode == 'gpe':
            self.embd = mlp(in_ch, out_ch, sigma)

    def forward(self, x):
        return self.embd(x)


