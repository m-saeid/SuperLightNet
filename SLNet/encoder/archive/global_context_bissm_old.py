import torch
import torch.nn as nn
import math

class RMSNorm(nn.Module):
    """
    Root‐Mean‐Square LayerNorm (no mean subtraction).
    Normalizes each token by its RMS, then applies scale γ.
    """
    def __init__(self, dim, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.scale

class BiSSMGlobalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1,
        use_rezero: bool = True,
        use_gate: bool = True,
        use_skip: bool = True,
        alpha_warmup_steps: int = 1000,
        low_rank: int = 4,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim

        # 1) RMSNorm instead of LayerNorm
        self.norm = RMSNorm(dim)

        self.hidden_dim = hidden_dim
        r = low_rank

        # -- Low-Rank + Skew-Symmetric A parameterization --
        # Diagonal stable core
        self.register_buffer(
            "A_diag",
            -torch.linspace(1.0, hidden_dim, hidden_dim) / hidden_dim
        )
        # Low‐rank factors
        self.U = nn.Parameter(torch.randn(hidden_dim, r) * 0.02)
        self.V = nn.Parameter(torch.randn(hidden_dim, r) * 0.02)
        # Skew‐symmetric matrix parameter
        self.S_param = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)

        # Input/output maps
        self.B = nn.Parameter(torch.randn(hidden_dim, dim) * (1.0 / math.sqrt(dim)))
        self.C = nn.Parameter(torch.randn(dim, hidden_dim) * (1.0 / math.sqrt(hidden_dim)))

        # Optional gating
        self.use_gate = use_gate
        if use_gate:
            self.gate_linear = nn.Linear(dim, dim)

        # ReZero α with warm-up schedule
        self.use_skip = use_skip
        self.alpha = nn.Parameter(torch.tensor(0.0))  # start at zero
        self._step = 0
        self.alpha_max = 0.1 if use_rezero and use_skip else 1.0
        self.warmup_steps = alpha_warmup_steps

        self.dropout = nn.Dropout(dropout)

    def _compute_A(self):
        # reconstruct A = D + U Vᵀ + skew(S_param)
        D = torch.diag(self.A_diag)
        UVt = self.U @ self.V.T
        S = self.S_param - self.S_param.T
        return D + UVt + S

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape

        # update α warm-up
        if self.training and self.use_skip and self.alpha_max > 0:
            self._step += 1
            frac = min(self._step / self.warmup_steps, 1.0)
            with torch.no_grad():
                self.alpha.fill_(self.alpha_max * frac)

        x_norm = self.norm(x)  # (B,N,D)
        A = self._compute_A()   # (H,H)

        # Forward recurrence
        h = x.new_zeros(B, self.hidden_dim)
        y_fwd = []
        for t in range(N):
            xt = x_norm[:, t, :]
            h = h @ A.T + xt @ self.B.T
            y_fwd.append((h @ self.C.T).unsqueeze(1))
        y_fwd = torch.cat(y_fwd, dim=1)

        # Backward recurrence
        h = x.new_zeros(B, self.hidden_dim)
        y_bwd = []
        for t in range(N - 1, -1, -1):
            xt = x_norm[:, t, :]
            h = h @ A.T + xt @ self.B.T
            y_bwd.append((h @ self.C.T).unsqueeze(1))
        y_bwd = torch.cat(list(reversed(y_bwd)), dim=1)

        # Combine & gate
        y = 0.5 * (y_fwd + y_bwd)
        if self.use_gate:
            g = torch.sigmoid(self.gate_linear(x_norm))
            y = y * g

        y = self.dropout(y)

        # Residual with ReZero α
        if self.use_skip:
            return x + self.alpha * y
        else:
            return y

class GlobalContextBiSSM(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int = None,
        n_layers: int = 1,
        **block_kwargs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            BiSSMGlobalBlock(dim, hidden_dim, **block_kwargs)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.blocks:
            out = block(out)
        return out

# Sanity‐check
if __name__ == '__main__':
    xyz = torch.rand(2, 128, 128)
    model = GlobalContextBiSSM(
        dim=128,
        hidden_dim=128,
        n_layers=2,
        dropout=0.1,
        use_rezero=True,
        use_gate=True,
        use_skip=True,
        alpha_warmup_steps=500,
        low_rank=8,
    )
    print("Total params:", sum(p.numel() for p in model.parameters()))
    out = model(xyz)
    print("Output shape:", out.shape)
