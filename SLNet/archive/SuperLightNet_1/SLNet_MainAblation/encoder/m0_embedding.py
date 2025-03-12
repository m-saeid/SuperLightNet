import torch
import torch.nn as nn
import torch.nn.functional as F
import math
try:
    from encoder.encoder_util import *
except:
    from encoder_util import *


class Embedding(nn.Module):
    def __init__(self, in_ch, out_ch, mode, sigma=0.3):
        super(Embedding, self).__init__()
        # self.in_ch = in_ch
        # self.out_ch = out_ch
        # self.mode = mode
        sigma = 0.3
        alpha = 100.0
        beta = 1.0

        # mlp, fourier, scaled_fourier, gaussian, harmonic, mlp2, learnable, relative, linear_coord
        input_dim, output_dim = in_ch, out_ch

        if mode == 'mlp':
            self.embd = mlp(input_dim, output_dim)
        elif mode == 'fourier':
            self.embd = FourierPositionalEncoding(input_dim, output_dim, num_frequencies=16, scale=1.0)
        elif mode == 'scaled_fourier':
            self.embd = ScaledFourierPositionalEncoding(input_dim, output_dim, alpha, beta)
        elif mode == 'gaussian':
            self.embd = GaussianPositionalEncoding(input_dim, output_dim, sigma)  
        elif mode == 'harmonic':
            self.embd = HarmonicPositionalEncoding(input_dim, output_dim, num_frequencies=4)
        elif mode == 'mlp2':
            self.embd = MLPPositionalEncoding(input_dim, output_dim, hidden_dim=64)
        elif mode == 'learnable':
            self.embd = LearnablePositionalEmbedding(num_points, output_dim) 
        elif mode == 'relative':
            self.embd = RelativePositionalEncoding(input_dim, output_dim, hidden_dim=64)
        elif mode == 'linear_coord':
            self.embd = LinearCoordinateEmbedding(input_dim, output_dim)
        elif mode == 'pointhop':
            self.embd = None
        else:
            raise Exception(f"transfer_mode!!! {mode}")

    def forward(self, x):
        return self.embd(x) if self.embd else x


