import torch
import torch.nn as nn

try:
    from encoder.encoder_util import mlp
except:
    from encoder_util import mlp

class Embedding(nn.Module):
    def __init__(self, in_ch, out_ch, mode, alpha_beta="yes_ba"):
        super(Embedding, self).__init__()
        self.alpha_beta = alpha_beta

        if mode == 'mlp':
            self.embd = mlp(in_ch, out_ch)
        else:
            raise Exception(f"embd_mode!!! {mode}")

        if alpha_beta=="yes_ab" or alpha_beta=="yes_ba":
            self.alpha = nn.Parameter(torch.ones([1, out_ch, 1]))
            self.beta = nn.Parameter(torch.zeros([1, out_ch, 1]))

    def forward(self, x):   # [B, C, N] 8,3,124
        x = self.embd(x) if self.embd else x    # x: [B, D, N] 
        if self.alpha_beta=="yes_ab":
            x = self.alpha * x + self.beta
        if self.alpha_beta=="yes_ba":
            x = self.alpha * (x + self.beta)
        return x