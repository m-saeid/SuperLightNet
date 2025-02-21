
class TE(nn.Module):
    """
    Unified Positional Encoding Module.

    Given an input tensor of shape [B, in_dim, N], this module computes Fourier-like
    positional features and projects them to an output tensor of shape [B, out_dim, N].

    Parameters:
      in_dim (int): Number of input channels (e.g. 3 for xyz coordinates).
      out_dim (int): Desired number of output channels.
      alpha (float): Base for frequency scaling.
      beta (float): Multiplicative factor for input scaling.
      feat_dim (int): Number of frequency components per input dimension.
                        The raw Fourier features will have size (in_dim * 2 * feat_dim).
    """
    def __init__(self, in_dim, out_dim, alpha, beta, feat_dim=16):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.beta = beta
        self.feat_dim = feat_dim
        
        # Compute the raw feature dimension: for each input dimension we generate sin and cos features.
        raw_feature_dim = in_dim * 2 * feat_dim
        # If the raw feature dimension does not equal out_dim, use a 1x1 conv (linear projection) to map to out_dim.
        if raw_feature_dim != out_dim:
            self.proj = nn.Conv1d(raw_feature_dim, out_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """
        Forward pass.
        
        Args:
          x (torch.Tensor): Input tensor of shape [B, in_dim, N].
        
        Returns:
          torch.Tensor: Positional encoding of shape [B, out_dim, N].
        """
        B, in_dim, N = x.shape
        device = x.device

        # Create a frequency range vector [0, 1, ..., feat_dim-1]
        feat_range = torch.arange(self.feat_dim, device=device).float()  # [feat_dim]
        # Compute frequency scales for each frequency: shape [1, 1, 1, feat_dim]
        dim_embed = torch.pow(self.alpha, feat_range / self.feat_dim).view(1, 1, 1, self.feat_dim)

        # Expand input x from [B, in_dim, N] to [B, in_dim, N, 1] to prepare for broadcasting
        x_expanded = x.unsqueeze(-1)
        # Scale the inputs and divide by the frequency scales: shape [B, in_dim, N, feat_dim]
        div_embed = (self.beta * x_expanded) / dim_embed

        # Compute sin and cos features
        sin_embed = torch.sin(div_embed)  # [B, in_dim, N, feat_dim]
        cos_embed = torch.cos(div_embed)  # [B, in_dim, N, feat_dim]

        # Concatenate sin and cos along the last dimension: [B, in_dim, N, 2*feat_dim]
        pos_embed = torch.cat([sin_embed, cos_embed], dim=-1)

        # Rearrange so that the Fourier features are in the channel dimension:
        # From [B, in_dim, N, 2*feat_dim] to [B, in_dim, 2*feat_dim, N], then flatten to [B, in_dim*2*feat_dim, N]
        pos_embed = pos_embed.permute(0, 1, 3, 2).reshape(B, in_dim * 2 * self.feat_dim, N)

        # Project to the desired output dimension if needed
        pos_embed = self.proj(pos_embed)  # [B, out_dim, N]

        return pos_embed

# Example usage
if __name__ == '__main__':
    # Example dimensions and hyperparameters
    batch_size = 8
    input_dim = 3          # e.g. xyz coordinates
    num_points = 1024
    output_dim = 64        # can be any number; not required to be input_dim * 2 * feat_dim
    alpha = 100.0
    beta = 1.0
    feat_dim = 16          # you can adjust this based on how many frequency components you want

    # Create random input: [batch, input_dim, num_points]
    x = torch.randn(batch_size, input_dim, num_points)

    # Initialize the positional encoding module
    pos_encoder = TE(input_dim, output_dim, alpha, beta, feat_dim)

    # Get the encoded features: [batch, output_dim, num_points]
    pos_features = pos_encoder(x)
    print("Output shape:", pos_features.shape)




import torch
import torch.nn as nn
import math

class GE(nn.Module):
    def __init__(self, in_dim, out_dim, sigma):
        """
        Gaussian Encoding (GE)
        Args:
            in_dim (int): Number of input channels (e.g., 3 for xyz).
            out_dim (int): Desired output dimension for the positional encoding.
            sigma (float): Standard deviation used in the RBF.
        """
        super(GE, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        
        # Compute number of RBF features per channel.
        feat_dim = math.ceil(out_dim / in_dim)
        self.feat_dim = feat_dim
        self.feat_num = feat_dim * in_dim
        
        # Create index selection to reduce the full RBF features to out_dim features.
        out_idx = torch.linspace(0, self.feat_num - 1, out_dim).long()
        self.register_buffer('out_idx', out_idx)
        
        # Create RBF centers for each channel.
        # We sample feat_dim values in (-1, 1) (excluding endpoints).
        feat_val = torch.linspace(-1.0, 1.0, feat_dim + 2)[1:-1]  # shape: [feat_dim]
        self.register_buffer('feat_val', feat_val.unsqueeze(0))  # shape: [1, feat_dim]

    def forward(self, xyz):
        """
        Args:
            xyz (torch.Tensor): Input tensor with shape either:
                                [B, N, in_dim] or [B, S, K, in_dim] or
                                [B, in_dim, N] or [B, in_dim, S, K].
        Returns:
            torch.Tensor: Positional encoding of shape [B, out_dim, N] or [B, out_dim, S, K].
        """
        # Adjust input shape so that channels are the last dimension.
        if xyz.dim() == 3:
            # Either [B, N, in_dim] or [B, in_dim, N]
            if xyz.shape[-1] != self.in_dim:
                xyz = xyz.permute(0, 2, 1)  # Now [B, N, in_dim]
        elif xyz.dim() == 4:
            # Either [B, S, K, in_dim] or [B, in_dim, S, K]
            if xyz.shape[-1] != self.in_dim:
                xyz = xyz.permute(0, 2, 3, 1)  # Now [B, S, K, in_dim]
        else:
            raise ValueError("Input must be either 3D or 4D tensor.")

        # Now assume xyz shape is [..., in_dim] where ... represents spatial dimensions.
        # Vectorized RBF computation:
        # Expand input to [..., in_dim, 1] and subtract the centers [1, feat_dim].
        # The broadcasting yields shape [..., in_dim, feat_dim].
        x_expanded = xyz.unsqueeze(-1)  # shape: [..., in_dim, 1]
        diff = x_expanded - self.feat_val  # shape: [..., in_dim, feat_dim]
        
        # Compute RBF features: exp(-0.5 * (difference/sigma)^2)
        rbf = torch.exp(-0.5 * (diff ** 2) / (self.sigma ** 2))  # [..., in_dim, feat_dim]
        
        # Flatten the last two dimensions to get shape [..., in_dim * feat_dim]
        rbf = rbf.view(*rbf.shape[:-2], self.feat_num)
        
        # Select out_dim features from the full set using precomputed indices.
        rbf_selected = torch.index_select(rbf, -1, self.out_idx)
        # Now rbf_selected has shape [..., out_dim]
        
        # Permute the dimensions so that channel comes before spatial dims.
        # For a 3D input [B, N, out_dim] -> [B, out_dim, N].
        # For a 4D input [B, S, K, out_dim] -> [B, out_dim, S, K].
        if rbf_selected.dim() == 3:
            out = rbf_selected.permute(0, 2, 1)
        elif rbf_selected.dim() == 4:
            out = rbf_selected.permute(0, 3, 1, 2)
        else:
            out = rbf_selected

        return out

# Example usage:
if __name__ == '__main__':
    batch_size = 128
    num_points = 1024
    in_dim = 3
    out_dim = 6   # Can be any number, even equal to in_dim.
    sigma = 0.3

    # Example input [B, N, in_dim]
    xyz = torch.randn(batch_size, num_points, in_dim)
    encoder = GE(in_dim, out_dim, sigma)
    pos_embed = encoder(xyz)  # Expected shape: [B, out_dim, N]
    print("Positional embedding shape:", pos_embed.shape)



import torch
import torch.nn as nn
import math

###############################################################################
# 1. Fourier (Sinusoidal) Encoding
###############################################################################
class FourierEncoding(nn.Module):
    """
    Computes Fourier (sinusoidal) features for each coordinate channel independently.
    
    For each input channel, this module computes:
      sin(2π * B * x) and cos(2π * B * x)
    where B is a randomly sampled frequency vector (per input channel).
    
    If the raw feature dimension (input_dim * 2 * num_frequencies) differs from
    the desired output_dim, a 1x1 convolution projects the features.
    """
    def __init__(self, input_dim, output_dim, num_frequencies=16, scale=1.0):
        super(FourierEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        # Create frequencies per input channel. Shape: [input_dim, num_frequencies]
        self.register_buffer('B', torch.randn(input_dim, num_frequencies) * scale)
        raw_feature_dim = input_dim * 2 * num_frequencies
        if raw_feature_dim != output_dim:
            self.proj = nn.Conv1d(raw_feature_dim, output_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
    def forward(self, x):
        # x: [B, input_dim, N]
        batch, input_dim, num_points = x.shape
        # Expand x to shape: [B, input_dim, N, 1]
        x_expanded = x.unsqueeze(-1)
        # Expand frequency tensor to shape: [1, input_dim, 1, num_frequencies]
        B_expanded = self.B.unsqueeze(0).unsqueeze(2)
        # Compute projections: [B, input_dim, N, num_frequencies]
        projection = 2 * math.pi * x_expanded * B_expanded
        sin_features = torch.sin(projection)
        cos_features = torch.cos(projection)
        # Concatenate along the frequency dimension: [B, input_dim, N, 2*num_frequencies]
        features = torch.cat([sin_features, cos_features], dim=-1)
        # Reshape to [B, input_dim*2*num_frequencies, N]
        features = features.permute(0, 1, 3, 2).reshape(batch, -1, num_points)
        # Project to the desired output dimension if needed
        return self.proj(features)

###############################################################################
# 2. Learnable Positional Encoding
###############################################################################
class LearnablePositionalEncoding(nn.Module):
    """
    Creates a learnable positional embedding.
    
    The embedding is a parameter of shape [1, output_dim, number_of_points] that
    is added (or concatenated) to your point features.
    """
    def __init__(self, num_points, output_dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, output_dim, num_points))
        
    def forward(self, x):
        # x: [B, input_dim, N] is not used here directly; the positional encoding is independent.
        return self.pos_embedding

###############################################################################
# 3. Relative Positional Encoding
###############################################################################
class RelativePositionalEncoding(nn.Module):
    """
    Computes positional features based on the relative offset of each point from the centroid.
    
    A small MLP processes these offsets to generate features.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(RelativePositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        # x: [B, input_dim, N]
        # Compute centroid for each point cloud: [B, input_dim, 1]
        centroid = x.mean(dim=-1, keepdim=True)
        # Compute relative coordinates: [B, input_dim, N]
        rel = x - centroid
        # Process with MLP (transpose to shape [B, N, input_dim])
        rel = self.mlp(rel.transpose(1, 2))
        # Transpose back to [B, output_dim, N]
        return rel.transpose(1, 2)

###############################################################################
# 4. Harmonic Encoding
###############################################################################
class HarmonicEncoding(nn.Module):
    """
    Uses fixed harmonic (sin/cos) functions with linearly spaced frequencies.
    
    For each coordinate channel, sin and cos features are computed using a set of frequencies.
    A 1x1 convolution is applied if the raw feature dimension does not equal output_dim.
    """
    def __init__(self, input_dim, output_dim, num_frequencies=4):
        super(HarmonicEncoding, self).__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        # Linearly spaced frequencies between 1 and num_frequencies.
        self.register_buffer('freqs', torch.linspace(1.0, num_frequencies, num_frequencies))
        raw_feature_dim = input_dim * 2 * num_frequencies
        if raw_feature_dim != output_dim:
            self.proj = nn.Conv1d(raw_feature_dim, output_dim, kernel_size=1)
        else:
            self.proj = nn.Identity()
    
    def forward(self, x):
        # x: [B, input_dim, N]
        batch, input_dim, num_points = x.shape
        # Expand x: [B, input_dim, N, 1]
        x_expanded = x.unsqueeze(-1)
        # Expand frequencies: [1, 1, 1, num_frequencies]
        freqs = self.freqs.view(1, 1, 1, -1)
        # Compute projections: [B, input_dim, N, num_frequencies]
        projection = x_expanded * freqs
        sin_features = torch.sin(projection)
        cos_features = torch.cos(projection)
        features = torch.cat([sin_features, cos_features], dim=-1)
        features = features.permute(0, 1, 3, 2).reshape(batch, -1, num_points)
        return self.proj(features)

###############################################################################
# 5. MLP-based Coordinate Encoding
###############################################################################
class MLPPositionalEncoding(nn.Module):
    """
    Maps raw coordinates to a higher-dimensional embedding via a lightweight MLP.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(MLPPositionalEncoding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # x: [B, input_dim, N]
        # Transpose to [B, N, input_dim] for MLP processing.
        x = x.transpose(1, 2)
        x = self.mlp(x)
        # Transpose back to [B, output_dim, N]
        return x.transpose(1, 2)

###############################################################################
# Example Usage
###############################################################################
if __name__ == '__main__':
    batch_size = 8
    input_dim = 3         # e.g. xyz coordinates
    num_points = 1024
    output_dim = 64

    # Create random input tensor: [B, input_dim, N]
    x = torch.randn(batch_size, input_dim, num_points)

    # Initialize each encoding
    fourier_enc = FourierEncoding(input_dim, output_dim, num_frequencies=16, scale=1.0)
    learnable_enc = LearnablePositionalEncoding(num_points, output_dim)
    relative_enc = RelativePositionalEncoding(input_dim, output_dim, hidden_dim=64)
    harmonic_enc = HarmonicEncoding(input_dim, output_dim, num_frequencies=4)
    mlp_enc = MLPPositionalEncoding(input_dim, output_dim, hidden_dim=64)

    # Get encoded features
    pos_fourier = fourier_enc(x)    # [B, output_dim, N]
    pos_learnable = learnable_enc(x)  # [1, output_dim, N] (learnable embedding independent of x)
    pos_relative = relative_enc(x)    # [B, output_dim, N]
    pos_harmonic = harmonic_enc(x)    # [B, output_dim, N]
    pos_mlp = mlp_enc(x)              # [B, output_dim, N]

    print("Fourier Encoding:", pos_fourier.shape)
    print("Learnable Encoding:", pos_learnable.shape)
    print("Relative Encoding:", pos_relative.shape)
    print("Harmonic Encoding:", pos_harmonic.shape)
    print("MLP Encoding:", pos_mlp.shape)


