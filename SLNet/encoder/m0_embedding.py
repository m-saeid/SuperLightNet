import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from encoder_util import *


class Embedding(nn.Module):
    def __init__(self, in_ch, out_ch, mode, sigma=0.3):
        super(Embedding, self).__init__()
        # self.in_ch = in_ch
        # self.out_ch = out_ch
        # self.mode = mode
        if mode == 'mlp':
            self.embd = mlp(in_ch, out_ch)
        elif mode == 'tpe':
            self.embd = TPE(in_ch, out_ch, alpha=1000, beta=100)
        elif mode == 'gpe':
            self.embd = GPE(in_ch, out_ch, sigma)
        elif mode == 'pointhop':
            self.embd = None
    def forward(self, x):
        return self.embd(x) if self.embd else x


