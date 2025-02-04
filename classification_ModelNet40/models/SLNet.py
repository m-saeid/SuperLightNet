import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from util import get_activation, square_distance, index_points, remove_points, farthest_point_sample, edgeSampling, sampling, query_ball_point, knn_point
except:
    from models.util import get_activation, square_distance, index_points, remove_points, farthest_point_sample, edgeSampling, sampling, query_ball_point, knn_point


class SmlplingGrouping(nn.Module):                                                                #          k      %             %
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", details=[["edge", 0.1 , 0.75], ["fps", 0.25]], **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(SmlplingGrouping, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        self.details = details
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points): # 2,1024,3  2,1024,16
        B, N, C = xyz.shape         # 2,1024,3
        S = self.groups             # 512
        xyz = xyz.contiguous()  # 2,1024,3    xyz [btach, points, xyz]
    #   2,512,3   2,512,16          2,1024,3  2,1024,16  [fps,1]  512
        new_xyz, new_points = sampling(xyz, points, self.details , S)

        idx = knn_point(self.kneighbors, xyz, new_xyz)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)        # [b, s, k, c]  (2,512,24,3)
        grouped_points = index_points(points, idx)  # [b, s, k, c]  (2,512,24,16)
        if self.use_xyz:    # Treu
            grouped_points = torch.cat([grouped_points, grouped_xyz],dim=-1)  # [b, s, k, d+c] (2,512,24,19)
        if self.normalize is not None:      # True
            if self.normalize =="center":   # False
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize =="anchor":   # True
                mean = torch.cat([new_points, new_xyz],dim=-1) if self.use_xyz else new_points # True (2,512,19)=cat((2,512,16),(2,512,3))
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]    (2,512,1,19)
            std = torch.std((grouped_points-mean).reshape(B,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # [b,1,1,1]
            grouped_points = (grouped_points-mean)/(std + 1e-5) #  (2,512,24,19) = (2,512,24,19)/(2,1,1,1)
            grouped_points = self.affine_alpha*grouped_points + self.affine_beta # (2,512,24,19)=(1,1,1,19)*(2,512,24,19)+(1,1,1,19)
    #   (2,512,24,35)           (2,512,24,19) ,                   (2,512,24,16)                               ,
        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points  #2,512,3    2,512,24,35
    



class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)




class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)




class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels,  blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PreExtraction, self).__init__()
        in_channels = 3+2*channels if use_xyz else 2*channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, k, d = x.size()       # [b,s,k,d] 2,512,24,35
        x = x.permute(0, 1, 3, 2)   # [b,s,d,k] 2,512,35,24
        x = x.reshape(-1, d, k)     # [b*s,d,k] 1024,35,24
        x = self.transfer(x)        # [b*s,d,k] 1024,32,24   transfer: Conv1d(35>32)+BN+ReLU
        batch_size, _, _ = x.size() # [b*s,d,k] 1024,32,24
        x = self.operation(x)       # [b*s,d,k] 1024,32,24   RESIDUAL BLOCKS
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x                    #[b,d,s] 2,32,512




class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu'):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):           # (2,32,512)  [b, d, s]
        return self.operation(x)    # (2,32,512)  [b, d, s]  RESIDUAL BLOCKS




class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2],
                 details=[["edge", 0.1 , 0.75], ["fps", 0.25]], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.embedding = ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation)
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            # append local_grouper_list
            local_grouper = SmlplingGrouping(last_channel, anchor_points, kneighbor, use_xyz, normalize, details)  # [b,g,k,d]
            self.local_grouper_list.append(local_grouper)
            # append pre_block_list
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                             res_expansion=res_expansion,
                                             bias=bias, activation=activation, use_xyz=use_xyz)
            self.pre_blocks_list.append(pre_block_module)

            # append pos_block_list
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                            res_expansion=res_expansion, bias=bias, activation=activation)
            self.pos_blocks_list.append(pos_block_module)

            last_channel = out_channel

        self.act = get_activation(activation)
        if last_channel > 128:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 512),
                nn.BatchNorm1d(512),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(256, self.class_num)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 128),
                nn.BatchNorm1d(128),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                self.act,
                nn.Dropout(0.5),
                nn.Linear(64, self.class_num)
            )# Give xyz[b, p, 3] and fea[b, p, d], return new_xyz[b, g, 3] and new_fea[b, g, k, d]

    def forward(self, x):            # [b,c,n] (2,3,1024)
        xyz = x.permute(0, 2, 1)     # [b,n,c] (2,1024,3)
        batch_size, _, _ = x.size()  # [b,c,n] (2,3,1024)
        x = self.embedding(x)        # [b,d,n] (2,16,1024)   MLP(3>16)   
        for i in range(self.stages): # i: 1,2,3,4
            # xyz: [b,       n>s       , 3]   x: [b,       d     ,       n>s       ]
            # xyz: (2, 1024:512:256:128, 3)   x: (2, 16:32:64:128,1024:512, 256:128)
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # xyz: [b,n>s,3] (2, 512:256:128:64, 3)    x: [b,n>s,k,d] (2, 512:256:128:64, 24, 35:67:131:259)
            x = self.pre_blocks_list[i](x) # +MaxPool                     #: [b, d, n>s] (2, 32:64:128:128, 512:256:128:64)
            x = self.pos_blocks_list[i](x)                                #: [b, d, n>s] (2, 32:64:128:128, 512:256:128:64)

        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) # [b,d,s]>[b,d]   (2,128,64)>(2,128)
        x = self.classifier(x)                          # [b,d]>[b,cls]   (2,128)>(128>128>64>40)
        return x                                        # [b,cls]         (2,40)
    

def SLNet(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["fps", 1.0]], **kwargs)


if __name__ == '__main__':
    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 1024)
    print("===> testing Model ...")
    model = SLNet()
    print(f'number of params: {trainable_params(model)}')
    out = model(data)
    print(out.shape)


