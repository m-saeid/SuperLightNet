import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fft import fft, ifft
import math

class SpectralFrustumAdapter(nn.Module):
    def __init__(self, dim: int, bottleneck: int = 32, curve_bits: int = 10, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.bottleneck = bottleneck
        self.bits = curve_bits

        self.down = nn.Linear(dim + 1, bottleneck)
        self.act  = nn.ReLU()
        r = max(1, bottleneck // 8)
        self.U = nn.Parameter(torch.randn(bottleneck, r) * 0.1)
        self.V = nn.Parameter(torch.randn(bottleneck, r) * 0.1)
        self.logit_d = nn.Parameter(torch.ones(bottleneck) * 4.0)
        self.up = nn.Linear(bottleneck, dim)
        self.dropout = nn.Dropout(dropout)

    def morton3D(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        max_val = 2**self.bits - 1
        xv = (x * max_val).long().clamp(0, max_val)
        yv = (y * max_val).long().clamp(0, max_val)
        zv = (z * max_val).long().clamp(0, max_val)
        def interleave(v):
            v = v & 0x1FFFFF
            v = v | (v << 32)
            v = v & 0x1F00000000FFFF
            v = v | ((v << 16) & 0x1F0000FF0000FF)
            v = v | ((v << 8)  & 0x100F00F00F00F)
            v = v | ((v << 4)  & 0x10C30C30C30C3)
            v = v | ((v << 2)  & 0x1249249249249)
            return v
        return interleave(xv) | (interleave(yv) << 1) | (interleave(zv) << 2)

    def spectral_filter(self, z: torch.Tensor) -> torch.Tensor:
        Zf = fft(z, dim=1)
        N = Zf.shape[1]
        w = torch.linspace(0, 2*math.pi, N, device=z.device)
        d = self.logit_d.clamp(-10.0, 10.0)
        H = torch.exp(-w.view(N,1) * d.view(1, -1))
        Yf = Zf * H.unsqueeze(0)
        y = ifft(Yf, dim=1).real
        mix = self.U @ self.V.T
        return y @ mix

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        morton_idx = self.morton3D(coords[...,0], coords[...,1], coords[...,2])
        order = morton_idx.argsort(dim=-1)
        x_perm = torch.gather(x, 1, order.unsqueeze(-1).expand(-1,-1,D))
        r = coords.norm(dim=-1, keepdim=True)
        z = torch.cat([x_perm, r], dim=-1)
        z = self.down(z)
        z = self.act(z)
        z = self.spectral_filter(z)
        out = self.up(z)
        return x + self.dropout(out)

# Test harness
if __name__ == "__main__":
    torch.manual_seed(0)

    # Hyperparameters
    B, N, D = 2, 128, 64
    bottleneck = 16

    # Dummy inputs
    feats = torch.randn(B, N, D, requires_grad=True)
    coords = torch.rand(B, N, 3)

    # Instantiate adapter
    adapter = SpectralFrustumAdapter(dim=D, bottleneck=bottleneck)

    # Forward
    out = adapter(feats, coords)
    print("Output shape:", out.shape)  # expect [B, N, D]

    # Gradient check
    loss = out.pow(2).mean()
    loss.backward()
    print("Grad on feats exists:", feats.grad is not None, 
          "sum:", feats.grad.abs().sum().item())
