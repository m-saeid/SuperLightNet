import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_util import *

                ##########################################################
                ###################### ResBlocks #########################
                ##########################################################


class Block2(nn.Module):
    def __init__(self, channels, blocks=1, res_dim_ratio=1, bias=True):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(Residual(channels, res_dim_ratio=res_dim_ratio, bias=bias))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):           # (2,32,512)  [b, d, s]
        return self.operation(x)    # (2,32,512)  [b, d, s]  RESIDUAL BLOCKS