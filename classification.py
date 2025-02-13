import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.Encoder import Encoder
from util import Classifier


class Classification(nn.Module):
        def __init__(self):
            super(Classification, self).__init__()
            self.encoder = Encoder(n=1024, embed=[3,16,'mlp'], res_dim_ratio=0.25, bias=False, use_xyz=True, norm_mode="anchor",
                            dim_ratio=[2, 2, 2, 1], num_blocks1=[1, 1, 2, 1], num_blocks2=[1, 1, 2, 1],
                            k_neighbors=[24,24,24,24], sampling_mode=['fps', 'fps', 'fps', 'fps'],
                            sampling_ratio=[2, 2, 2, 2])
            self.classifier = Classifier(128, 40)  # 128 = embed[1]*dim_ratios = 16*2*2*2*1 
        
        def forward(self, x):
             xyz_list, f_list = self.encoder(x)
             x = F.adaptive_max_pool1d(f_list[-1], 1).squeeze(dim=-1)
             return self.classifier(x)
             

if __name__ == '__main__':
    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 1024)
    print("===> testing Model ...")
    model = Classification()
    print(f'number of params: {trainable_params(model)}')
    out = model(data)
    print(out.shape)





"""
class args:
    n=1024
    num_cls=40
    embed_dim=16
    res_expansion=0.25,
    bias=False
    use_xyz=True
    norm="anchor"
    dim_expansion=[2, 2, 2, 1]
    pre_blocks=[1, 1, 2, 1]
    pos_blocks=[1, 1, 2, 1]
    k_neighbors=[24,24,24,24]
    reducers=[2, 2, 2, 2],

###################
# Model(args) #######################################
###################
"""