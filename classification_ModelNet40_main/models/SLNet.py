import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from util import get_activation, square_distance, index_points, remove_points, farthest_point_sample, edgeSampling, sampling, query_ball_point, knn_point
except:
    from models.util import get_activation, square_distance, index_points, remove_points, farthest_point_sample, edgeSampling, sampling, query_ball_point, knn_point


                ##########################################################
                ###################### Functions #########################
                ##########################################################

def sampling(xyz, s, mode):
    if mode == "fps":
        idx = farthest_point_sample(xyz, s).long()
    return idx


def grouping(k, radius, xyz, new_xyz, mode):
    if mode == "knn":
        idx = knn_point(k, xyz, new_xyz)  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
    elif mode == "ball":
        idx = query_ball_point(radius, k, xyz, new_xyz)
    return idx


def normalize(norm, f_grouped, f_sampled, xyz_sampled, use_xyz):
    b,_,_ = xyz_sampled.shape
    if norm =="center":   # False
        mean = torch.mean(f_grouped, dim=2, keepdim=True)
    if norm =="anchor":   # True
        mean = torch.cat([f_sampled, xyz_sampled],dim=-1) if use_xyz else f_sampled # True (2,512,19)=cat((2,512,16),(2,512,3))
        mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]    (2,512,1,19)
    std = torch.std((f_grouped-mean).reshape(b,-1),dim=-1,keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1) # [b,1,1,1]
    f_grouped = (f_grouped-mean)/(std + 1e-5) #  (2,512,24,19) = (2,512,24,19)/(2,1,1,1)
    return f_grouped


def res_pooling(x, b, n, mode="adaptive_max_pool1d"):
    batch_size = b*n
    if mode == "adaptive_max_pool1d":
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x


def final_pooling(x, mode="adaptive_max_pool1d"):
    if mode == "adaptive_max_pool1d":
        return F.adaptive_max_pool1d(x, 1).squeeze(dim=-1) # [b,d,s]>[b,d]   (2,128,64)>(2,128)


                ##########################################################
                ####################### Classes ##########################
                ##########################################################


class mlp(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super(mlp, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class Residual(nn.Module):
    def __init__(self, ch, res_exp=1.0, bias=True):
        super().__init__()
        self.act = nn.ReLU(inplace=True)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=ch, out_channels=int(ch * res_exp),
                      kernel_size=1, groups=1, bias=bias),
            nn.BatchNorm1d(int(ch * res_exp)),
            nn.ReLU(inplace=True)
        )
        self.net2 = nn.Sequential(
            nn.Conv1d(in_channels=int(ch * res_exp), out_channels=ch,
                      kernel_size=1, bias=bias),
            nn.BatchNorm1d(ch)
        )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)


                ##########################################################
                ##################### LocalGrouper #######################
                ##########################################################

class LocalGrouper(nn.Module):                                                                #          k      %             %
    def __init__(self, d, s, k, use_xyz=True, norm="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param s: s number
        :param k: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.s = s
        self.k = k
        self.use_xyz = use_xyz
        if norm is not None:
            self.norm = norm.lower()
        else:
            self.norm = None
        if self.norm not in ["center", "anchor"]:
            print(f"Unrecognized norm parameter (self.norm), set to None. Should be one of [center, anchor].")
            self.norm = None
        if self.norm is not None:
            add_d=3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1,1,1,d + add_d]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, d + add_d]))

    def forward(self, xyz, f): # 2,1024,3  2,1024,16
        B, N, C = xyz.shape         # 2,1024,3
        xyz = xyz.contiguous()  # 2,1024,3    xyz [btach, n, xyz]

        # SAMPLING
        idx = sampling(xyz, self.s, mode="fps")
        xyz_sampled = index_points(xyz, idx)
        f_sampled = index_points(f, idx)

        # GROPPING
        idx = grouping(self.k, 0, xyz, xyz_sampled, mode="knn")  # (2,512,24) = knn(24, (2,1024,3), (2,512,3))
        xyz_grouped = index_points(xyz, idx)        # [b, s, k, c]  (2,512,24,3)
        f_grouped = index_points(f, idx)  # [b, s, k, c]  (2,512,24,16)

        # USE_XYZ
        f_grouped = torch.cat([f_grouped, xyz_grouped],dim=-1) if self.use_xyz else f_grouped
            
        # NORMALIZE
        if self.norm is not None:
            f_grouped = normalize(self.norm, f_grouped, f_sampled, xyz_sampled, self.use_xyz)
            f = self.affine_alpha*f_grouped + self.affine_beta

        f_sampled = torch.cat([f_grouped, f_sampled.view(B, self.s, 1, -1).repeat(1, 1, self.k, 1)], dim=-1)
        return xyz_sampled, f_sampled  #2,512,3    2,512,24,35
    

                ##########################################################
                ##################### RedResBlock ########################
                ##########################################################

class RedResBlock(nn.Module):    # reduce d  and  residual
    def __init__(self, ch, out_ch,  blocks=1, res_expansion=1, bias=True,
                 use_xyz=True):
        """
        input: [b,g,k,d]: output:[b,d,g]
        :param ch:
        :param blocks:
        """
        super().__init__()
        in_ch = 3+2*ch if use_xyz else 2*ch
        self.transfer = mlp(in_ch, out_ch, bias=bias)
        operation = []
        for _ in range(blocks):
            operation.append(
                Residual(out_ch, res_exp=res_expansion,
                                bias=bias)
            )
        self.operation = nn.Sequential(*operation)

    def forward(self, x):
        b, n, k, d = x.size()       # [b,s,k,d] 2,512,24,35
        x = x.permute(0, 1, 3, 2)   # [b,s,d,k] 2,512,35,24
        x = x.reshape(-1, d, k)     # [b*s,d,k] 1024,35,24
        x = self.transfer(x)        # [b*s,d,k] 1024,32,24   transfer: Conv1d(35>32)+BN+ReLU
        #batch_size, _, _ = x.size() # [b*s,d,k] 1024,32,24
        x = self.operation(x)       # [b*s,d,k] 1024,32,24   RESIDUAL BLOCKS
        # x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1) # (k) [b*s,d] 1024,32
        # x = x.reshape(b, n, -1).permute(0, 2, 1)             #     [b,d,s] 2,32,512
        return x          # 1024,32,24          #[b,d,s] 2,32,512
    

                ##########################################################
                ###################### ResBlocks #########################
                ##########################################################


class ResBlocks(nn.Module):
    def __init__(self, channels, blocks=1, res_expansion=1, bias=True):
        """
        input[b,d,g]; output[b,d,g]
        :param channels:
        :param blocks:
        """
        super().__init__()
        operation = []
        for _ in range(blocks):
            operation.append(Residual(channels, res_exp=res_expansion, bias=bias))
        self.operation = nn.Sequential(*operation)

    def forward(self, x):           # (2,32,512)  [b, d, s]
        return self.operation(x)    # (2,32,512)  [b, d, s]  RESIDUAL BLOCKS


                ##########################################################
                ###################### Classifier ########################
                ##########################################################

class Classifier(nn.Module):
    def __init__(self, last_channel, num_cls):
        super(Classifier, self).__init__()
        if last_channel > 128:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_cls)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_cls)
            )
    def forward(self, x):
        return self.classifier(x)


                ##########################################################
                ######################### Model ##########################
                ##########################################################


class Model(nn.Module):
    def __init__(self, n=1024, num_cls=40, embed_dim=64, res_expansion=1.0,
                 bias=True, use_xyz=True, norm="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], **kwargs):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = num_cls
        self.points = n
        self.embedding = mlp(3, embed_dim, bias=bias)
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
            self.local_grouper_list.append(LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, norm))  # [b,g,k,d])
            self.pre_blocks_list.append(RedResBlock(last_channel, out_channel, pre_block_num, res_expansion=res_expansion,
                                             bias=bias, use_xyz=use_xyz))
            self.pos_blocks_list.append(ResBlocks(out_channel, pos_block_num, res_expansion=res_expansion, bias=bias))

            last_channel = out_channel

        #self.classifier = Classifier(last_channel, self.class_num) ####################################################
        if last_channel > 128:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, num_cls)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(last_channel, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(64, num_cls)
            )


    def forward(self, x):            # [b,c,n] (2,3,1024)
        xyz = x.permute(0, 2, 1)     # [b,n,c] (2,1024,3)
        batch_size, _, _ = x.size()  # [b,c,n] (2,3,1024)
        x = self.embedding(x)        # [b,d,n] (2,16,1024)   MLP(3>16)   
        for i in range(self.stages): # i: 1,2,3,4
            # xyz: [b,       n>s       , 3]   x: [b,       d     ,       n>s       ]
            # xyz: (2, 1024:512:256:128, 3)   x: (2, 16:32:64:128,1024:512, 256:128)
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))  # xyz: [b,n>s,3] (2, 512:256:128:64, 3)    x: [b,n>s,k,d] (2, 512:256:128:64, 24, 35:67:131:259)
            b, n, _ = xyz.shape
            x = self.pre_blocks_list[i](x) # +MaxPool                     #: [b, d, n>s] (2, 32:64:128:128, 512:256:128:64)
            x = res_pooling(x, b, n, mode="adaptive_max_pool1d")          #     [b,d,s] 2,32,512
            x = self.pos_blocks_list[i](x)                                #: [b, d, n>s] (2, 32:64:128:128, 512:256:128:64)

        x = final_pooling(x, mode="adaptive_max_pool1d") # [b,d,s]>[b,d]   (2,128,64)>(2,128)
        x = self.classifier(x)                           # [b,d]>[b,cls]   (2,128)>(128>128>64>40)
        return x                                         # [b,cls]         (2,40)
    


                ##########################################################
                ##########################################################
                ##########################################################




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





def SLNet(num_cls=40, **kwargs) -> Model:
    return Model(n=1024, num_cls=num_cls, embed_dim=16, res_expansion=0.25,
                    bias=False, use_xyz=True, norm="anchor",
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    **kwargs)




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


