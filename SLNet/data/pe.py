import torch
import torch.nn as nn
import torch.nn.functional as F
import math 


class AdaptiveEncoding(nn.Module):
    """
    Adaptive Embedding Function (AdaptiveEmbedding)
    
    This function implements an adaptive, data-driven variant of the standard
    Gaussian (RBF) embedding. It adjusts the kernel width (sigma) based on the 
    global standard deviation of the input and uses an adaptive blending strategy
    to fuse the Gaussian response with a complementary cosine response.
    
    Assumptions and Implementation Details:
    
    1. Adaptive Kernel Width:
       - Compute a global standard deviation from the input (over points) and 
         adjust the effective sigma as: adaptive_sigma = base_sigma * (1 + global_std).
         
    2. Adaptive Blending:
       - Compute a blend weight (between 0 and 1) as: 
           blend = sigmoid((global_std - baseline) * scaling)
         This weight is used to fuse the Gaussian (RBF) embedding with a cosine embedding.
         
    3. Dynamic Normalization:
       - The difference (tmp) is divided by the adaptive sigma to normalize the scale 
         of the kernel function.
    
    4. Complementarity:
       - The Gaussian captures local similarity via an exponential decay,
         whereas the cosine transformation introduces a periodic component.
         Their fusion is intended to yield a richer representation.
    
    5. Parameterlessness:
       - All adaptation is computed on-the-fly from the data, with no learnable parameters.
    
    Args:
      in_dim (int): Input dimension (typically 3 for XYZ coordinates).
      out_dim (int): Desired output dimension.
      sigma (float): Base sigma value (default kernel width).
      baseline (float): A fixed baseline for computing blend weight (default 0.1).
      scaling (float): Scaling factor for the sigmoid to compute blend (default 10.0).
      eps (float): Small constant to prevent division by zero.
    """
    def __init__(self, in_dim, out_dim, sigma=0.4, baseline=0.1, scaling=10.0, eps=1e-6):
        super(AdaptiveEncoding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.base_sigma = sigma  # base kernel width
        self.baseline = baseline
        self.scaling = scaling
        self.eps = eps
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        # Fixed grid of values for embedding (excluding endpoints)
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
        
    def forward(self, xyz):
        """
        Args:
          xyz: Tensor of shape [B, N, in_dim] or [B, S, K, in_dim]
        Returns:
          Tensor of shape [B, ..., out_dim] computed by adaptively fusing a Gaussian 
          and a cosine response.
        """

        if xyz.shape[-1] != 3:
            xyz = xyz.permute(0,2,1)

        if self.out_dim == 0:
            return xyz
        if xyz.dim() == 3:
            # Compute global standard deviation across points (dim=1)
            global_std = torch.mean(torch.std(xyz, dim=1))
        elif xyz.dim() == 4:
            # Reshape to [B, -1, in_dim] and compute standard deviation over points
            global_std = torch.mean(torch.std(xyz.view(xyz.size(0), -1, self.in_dim), dim=1))
        else:
            raise ValueError("Input must be 3D or 4D")
        
        # Adaptive sigma: scale the base sigma by (1 + global_std)
        adaptive_sigma = self.base_sigma * (1 + global_std)
        # Adaptive blend weight via sigmoid; yields a value in (0,1)
        blend = torch.sigmoid((global_std - self.baseline) * self.scaling)
        
        embeds = []
        for i in range(self.in_dim):
            # Compute difference from fixed grid values
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Gaussian (RBF) component using adaptive sigma
            rbf = (-0.5 * (tmp / (adaptive_sigma + self.eps))**2).exp()
            # Cosine component using the same adaptive sigma for scaling
            cosine = torch.cos(tmp / (adaptive_sigma + self.eps))
            # Adaptive fusion of the two components:
            combined = blend * rbf + (1 - blend) * cosine
            embeds.append(combined)
        
        # Concatenate all channels and select the desired output dimensions
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed   #.permute(0,2,1)



# New Hybrid/Augmented Embedding Function:
class HybridEncoding(nn.Module):
    """
    Hybrid/Augmented Embedding Function
    
    Assumptions:
    1. Complementarity: Combines the RBF (Gaussian) embedding and a complementary cosine embedding.
       - RBF captures local Euclidean similarity.
       - Cosine highlights periodic/angle-related aspects.
    2. Parameterlessness: Fusion is achieved with fixed weights (a blend factor) without adding learnable parameters.
    3. Fusion Strategy: Both embeddings are computed to produce vectors of the same size.
       An element-wise weighted sum is then used to fuse them.
    4. Robustness: The combined representation is assumed to be richer, capturing more nuances.
    5. No Extra Learning Overhead: All operations are fixed functions.
    
    For each input channel, computes:
      - RBF component: exp(-0.5 * ((x - v)/sigma)^2)
      - Cosine component: cos(x - v)
    Fuses them as: output = blend * RBF + (1 - blend) * cosine.
    """
    def __init__(self, in_dim, out_dim, sigma=0.31, blend=0.5):
        super(HybridEncoding, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        self.blend = blend  # Fixed blending weight
        
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_num = feat_dim * in_dim
        self.out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        self.feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1].reshape(1, -1)
    
    def forward(self, xyz):

        if xyz.shape[-1] != 3:
            xyz = xyz.permute(0,2,1)
    
        if self.out_dim == 0:
            return xyz
        if xyz.dim() not in {3, 4}:
            raise ValueError("Input must be either [B, N, in_dim] or [B, S, K, in_dim]")
        if xyz.shape[-1] == 3:
            xyz.permute
        embeds = []
        for i in range(self.in_dim):
            tmp = xyz[..., i:i+1] - self.feat_val.to(xyz.device)
            # Compute RBF component (Gaussian)
            rbf = (-0.5 * tmp**2 / (self.sigma**2)).exp()
            # Compute complementary cosine component
            cosine = torch.cos(tmp)
            # Fuse them using fixed blending weights
            combined = self.blend * rbf + (1 - self.blend) * cosine
            embeds.append(combined)
        position_embed = torch.cat(embeds, dim=-1)
        position_embed = torch.index_select(position_embed, -1, self.out_idx.to(xyz.device))
        return position_embed   #.permute(0,2,1)