import torch
import torch.nn as nn

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from encoder.encodings import *

import torch
import torch.nn as nn
import math


                ##########################################################
                ###################### Functions #########################
                ##########################################################


'''
def sampling(xyz, s, mode):
    if mode == "fps":
        idx = farthest_point_sample(xyz, s).long()
    return idx
'''

def grouping(k, radius, xyz, new_xyz, mode):
    if mode == "knn":
        idx = knn_point(k, xyz, new_xyz)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
    elif mode == "ball":
        idx = query_ball_point(radius, k, xyz, new_xyz)
    return idx


def normalize(norm, f_grouped, f_sampled, xyz_sampled, use_xyz):
    b,_,_ = xyz_sampled.shape
    if norm =="center":   # False
        mean = torch.mean(f_grouped, dim=2, keepdim=True)
    if norm =="anchor":   # True
        mean = torch.cat([f_sampled, xyz_sampled],dim=-1) if use_xyz else f_sampled # True (2,512,19)=cat((2,512,16),(2,512,3))
        mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]    (2,512,1,19)
    std = torch.std((f_grouped-mean).reshape(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # [b,1,1,1]
    f_grouped = (f_grouped-mean)/(std + 1e-5) #  (2,512,24,19) = (2,512,24,19)/(2,1,1,1)
    return f_grouped


def res_pooling(x, b, n, mode="adaptive_max_pool1d"):
    batch_size = b*n
    if mode == "adaptive_max_pool1d":
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x


def final_pooling(x, mode="adaptive_max_pool1d"):
    if mode == "adaptive_max_pool1d":
        return F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) # [b,d,s]>[b,d]   (2,128,64)>(2,128)


                ##########################################################
                ####################### Classes ##########################
                ##########################################################

class mlp(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    def __init__(self, ch, res_mode, res_dim_ratio=1.0, bias=True):
        super().__init__()
        self.act = nn.ReLU(inplace=True)

        if res_mode == "mlp":
            self.net1 = nn.Sequential(
                nn.Conv1d(in_channels=ch, out_channels=int(ch * res_dim_ratio), kernel_size=1, groups=1, bias=bias),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(ch * res_dim_ratio), out_channels=ch, kernel_size=1, bias=bias),
                nn.BatchNorm1d(ch))
            
        elif res_mode == 'fourier':
            self.net1 = nn.Sequential(
                FourierPositionalEncoding(input_dim=ch, output_dim=int(ch * res_dim_ratio), num_frequencies=16, scale=1.0),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(ch * res_dim_ratio), out_channels=ch, kernel_size=1, bias=bias),
                FourierPositionalEncoding(input_dim=int(ch * res_dim_ratio), output_dim=ch, num_frequencies=16, scale=1.0),
                nn.BatchNorm1d(ch))
            
        elif res_mode == 'scaled_fourier':
            self.net1 = nn.Sequential(
                ScaledFourierPositionalEncoding(input_dim=ch, output_dim=int(ch * res_dim_ratio), alpha=100.0, beta=1.0),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(ch * res_dim_ratio), out_channels=ch, kernel_size=1, bias=bias),
                ScaledFourierPositionalEncoding(input_dim=int(ch * res_dim_ratio), output_dim=ch, alpha=100.0, beta=1.0),
                nn.BatchNorm1d(ch))
            
        elif res_mode == 'gaussian':
            self.net1 = nn.Sequential(
                GaussianPositionalEncoding(input_dim=ch, output_dim=int(ch * res_dim_ratio), sigma=0.3),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                GaussianPositionalEncoding(input_dim=int(ch * res_dim_ratio), output_dim=ch, sigma=0.3),
                nn.BatchNorm1d(ch))
            
        elif res_mode == 'harmonic':
            self.net1 = nn.Sequential(
                HarmonicPositionalEncoding(input_dim=ch, output_dim=int(ch * res_dim_ratio), num_frequencies=4),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                HarmonicPositionalEncoding(input_dim=int(ch * res_dim_ratio), output_dim=ch, num_frequencies=4),
                nn.BatchNorm1d(ch))
            
        elif res_mode == 'mlp2':
            self.net1 = nn.Sequential(
                MLPPositionalEncoding(input_dim=ch, output_dim=int(ch * res_dim_ratio), hidden_dim=64),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                MLPPositionalEncoding(input_dim=int(ch * res_dim_ratio), output_dim=ch, hidden_dim=64),
                nn.BatchNorm1d(ch))
  
        #elif transfer_mode == 'learnable':
         #   self.net1 = nn.Sequential(
         #       nn.Conv1d(in_channels=ch, out_channels=int(ch * res_dim_ratio), kernel_size=1, groups=1, bias=bias),
         #       LearnablePositionalEmbedding(num_points, output_dim),
         #       nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
         #   self.net2 = nn.Sequential(
         #       nn.Conv1d(in_channels=int(ch * res_dim_ratio), out_channels=ch, kernel_size=1, bias=bias),
         #       LearnablePositionalEmbedding(num_points, output_dim=ch) 
         #       nn.BatchNorm1d(ch))

        elif res_mode == 'relative':
            self.net1 = nn.Sequential(
                RelativePositionalEncoding(input_dim=ch, output_dim=int(ch * res_dim_ratio), hidden_dim=64),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                RelativePositionalEncoding(input_dim=int(ch * res_dim_ratio), output_dim=ch, hidden_dim=64),
                nn.BatchNorm1d(ch))
            
        elif res_mode == 'linear_coord':
            self.net1 = nn.Sequential(
                LinearCoordinateEmbedding(input_dim=ch, output_dim=int(ch * res_dim_ratio)),
                nn.BatchNorm1d(int(ch * res_dim_ratio)), nn.ReLU(inplace=True))
            self.net2 = nn.Sequential(
                LinearCoordinateEmbedding(input_dim=int(ch * res_dim_ratio), output_dim=ch),
                nn.BatchNorm1d(ch))
            
        else:
               raise Exception(f"res_mode!!! {res_mode}")

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)





def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def remove_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: indices of the points to remove, [B, S]
    Return:
        new_points: points data with specified indices removed, [B, N-S, C]
    """
    B, N, C = points.shape
    device = points.device
    
    # Create a mask for each batch
    mask = torch.ones((B, N), dtype=torch.bool).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(-1, 1)
    mask[batch_indices, idx] = False

    # Apply mask to remove points and reshape
    new_points = points[mask].view(B, -1, C)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def edgeSampling(xyz, k, s):
    """
    Input:
        s: sampling numbers
        xyz: all points, [B, N, C]
        k = number of neighbors to attention
    Return:
        samples_idx: sampled pointcloud index, [B, npoint]
    """
    number_of_k = int(xyz.shape[1] * k) # int(s * k)
    sqrdists = square_distance(xyz, xyz)    # 2,1024,1024
    sqrdists = -1*((-1*sqrdists).topk(number_of_k)[0])         # 2,1024,number_ok_k
    distances = sqrdists.sum(dim=-1)        # 2,1024
    idx = distances.topk(s, dim=-1)[1]      # 2,512
    return idx


def sampling(xyz, points, details, S):
    """
    Perform sampling based on details.
    
    Args:
        xyz: Input points, tensor of shape [B, N, 3]
        points: Additional points data, tensor of shape [B, N, d]
        details: List of sampling methods and their parameters
        S: Total number of samples to obtain

    Returns:
        new_xyz: Sampled xyz points, tensor of shape [B, S, 3]
        new_points: Sampled points data, tensor of shape [B, S, d]
    """
    N = xyz.shape[1]
    s = [int(d[-1] * S) for d in details]
    if sum(s) != S:
        diff = S - sum(s)
        s[-1] += diff

    new_xyz = []
    new_points = []
    
    for i in range(len(details)):
        if details[i][0] == "edge":
            idx = edgeSampling(xyz, details[i][1], s[i])
        elif details[i][0] == "fps":
            idx = farthest_point_sample(xyz, s[i]).long()
        else:
            raise ValueError(f"Unknown sampling method: {details[i][0]}")

        new_xyz.append(index_points(xyz, idx))
        new_points.append(index_points(points, idx))

        xyz = remove_points(xyz, idx)
        points = remove_points(points, idx)

    new_xyz = torch.cat(new_xyz, dim=1)
    new_points = torch.cat(new_points, dim=1)

    return new_xyz, new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx




"""
# PosE for Raw-point Embedding 
class TPE_old(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz): # B,C,N
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim * 2)
        
        feat_range = torch.arange(feat_dim).float().cuda() if torch.cuda.is_available() else torch.arange(feat_dim).float()    
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)

        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)
        
        return position_embed
    

class GPE_old(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        super(GPE_old, self).__init__()
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
"""