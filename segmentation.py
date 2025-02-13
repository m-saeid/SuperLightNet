import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.Encoder import Encoder
from encoder.encoder_util import mlp, square_distance, index_points
from encoder.m6_block2 import Block2


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, out_channel, blocks=1, res_expansion=1.0, bias=True):
        super(PointNetFeaturePropagation, self).__init__()
        self.fuse = mlp(in_channel, out_channel, bias=bias)
        #self.extraction = PosExtraction_(out_channel, blocks, groups=groups,
        #                                res_expansion=res_expansion, bias=bias, activation=activation)
        self.extraction = Block2(out_channel, blocks, res_dim_ratio=res_expansion, bias=bias)

    def forward(self, xyz1, xyz2, points1, points2):
        # B N 3   -  B S 3  -  B D' N    -  B D'' S
        # 2 32 3  -  2 8 3  -  2 128 32  -  2 128 8
        """
        Input:
            xyz1: input points position data, [B, N, 3]
            xyz2: sampled input points position data, [B, S, 3]
            points1: input points data, [B, D', N]
            points2: input points data, [B, D'', S]
        Return:
            new_points: upsampled points data, [B, D''', N]
        """
        # xyz1 = xyz1.permute(0, 2, 1)
        # xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1) # B S D''   2 8 128 
        B, N, C = xyz1.shape               # B N 3     2 32 3
        _, S, _ = xyz2.shape               # B S 3      2 8 3

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)  #  2 8 128 > 
        else:
            dists = square_distance(xyz1, xyz2)            # (B N 3, B S 3) > B N S  -  2 32 8
            dists, idx = dists.sort(dim=-1)                # B N S , B N S  -  2 32 8 , 2 32 8
            dists, idx = dists[:, :, :3], idx[:, :, :3]    # k=3  B N 3 , B N 3  -  2 32 3 - 2 32 3

            dist_recip = 1.0 / (dists + 1e-8)                   # B N 3  -  2 32 3
            norm = torch.sum(dist_recip, dim=2, keepdim=True)   # B N 1  -  2 32 1
            weight = dist_recip / norm                          # B N 3  -  2 32 3
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) # B N D''  2 32 128

        if points1 is not None:                     # True
            points1 = points1.permute(0, 2, 1)      # [B N D']   2 32 128
            new_points = torch.cat([points1, interpolated_points], dim=-1)  # [B N D"""]     2 32 256
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)    # B D""" N    2 256 32

        new_points = self.fuse(new_points)          #   B D* N > B D** N
        # MLP1(256>512; [2,256,32]>[2,512,32])     MLP2(576>256; [2,576,128]>[2,256,128])
        # MLP3(288>128; [2,288,512]>[2,128,512])   MLP4(144>128; [2,144,2048]>[2,128,2048])
        new_points = self.extraction(new_points)
        # 2 512 32 > 2 512 32  -  2 256 128 > 2 256 128  -  2 128 512 > 2 128 512  -  2 128 2048 > 2 128 2048
        return new_points
        # 2 512 32


"""
def LPTNet(num_classes=50, **kwargs) -> Model:
   return Model(num_classes=num_classes, points=2048, embed_dim=16, groups=1, res_expansion=1.0,
                activation="relu", bias=True, use_xyz=True, normalize="anchor",
                dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4], # [2,2,2,2]
                de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                gmp_dim=64,cls_dim=64, k_attention=[24,24,24,8], **kwargs)
"""



class Decoder(nn.Module):
    def __init__(self, de_dims=[512, 256, 128, 128], de_blocks=[2,2,2,2], gmp_dim=64,cls_dim=64, **kwargs):
        super(Decoder, self).__init__()

        self.encoder = Encoder(n=2048, embed=[6,16,'mlp'], res_dim_ratio=1, bias=False, use_xyz=True, norm_mode="anchor",
                                    dim_ratio=[2, 2, 2, 1], num_blocks1=[1, 1, 2, 1], num_blocks2=[1, 1, 2, 1],
                                    k_neighbors=[32,32,32,32], sampling_mode=['fps', 'fps', 'fps', 'fps'],
                                    sampling_ratio=[2, 2, 2, 2])

        '''
        return Model(num_classes=num_classes, points=2048, embed_dim=16, groups=1, res_expansion=1.0,
                        activation="relu", bias=True, use_xyz=True, normalize="anchor",
                        dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                        k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4], # [2,2,2,2]
                        de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                        gmp_dim=64,cls_dim=64, k_attention=[24,24,24,8], **kwargs)
        '''

        res_dim_ratio = 0.25
        num_classes = 50
        bias = False

        en_dims = [16,32,64,128,128]
        ### Building Decoder #####
        self.decode_list = nn.ModuleList()
        en_dims.reverse()
        de_dims.insert(0,en_dims[0])
        assert len(en_dims) == len(de_dims) == len(de_blocks)+1
        for i in range(len(en_dims)-1):
            self.decode_list.append(
                PointNetFeaturePropagation(de_dims[i]+en_dims[i+1], de_dims[i+1],
                                           blocks=de_blocks[i], res_expansion=res_dim_ratio,
                                           bias=bias)
            )

        # class label mapping
        self.cls_map = nn.Sequential(
            mlp(16, cls_dim, bias=bias),
            mlp(cls_dim, cls_dim, bias=bias)
        )
        # global max pooling mapping
        self.gmp_map_list = nn.ModuleList()
        for en_dim in en_dims:
            self.gmp_map_list.append(mlp(en_dim, gmp_dim, bias=bias))
        self.gmp_map_end = mlp(gmp_dim*len(en_dims), gmp_dim, bias=bias)

        # classifier
        self.classifier = nn.Sequential(
            nn.Conv1d(gmp_dim+cls_dim+de_dims[-1], 128, 1, bias=bias),
            nn.BatchNorm1d(128),
            nn.Dropout(),
            nn.Conv1d(128, num_classes, 1, bias=bias)
        )
        self.en_dims = en_dims

    def forward(self, x, norm_plt, cls_label):
        xyz = x                                     # [B, C, N]  (2,3,2048)
        x = torch.cat([x, norm_plt],dim=1)          # [B, 6, N]  (2,6,2048)

        xyz_list, x_list = self.encoder(xyz, x)
        # xyz_list                                  # [B, S, C]  (2, 2048,1024,512,256,128, 3)
        # x_list                                    # [B, D, S]  (2, 16,32,64,128,128, 2048,1024,512,256,128)

        # Decoder
        xyz_list.reverse()              # len = 5     [B, S, C]  (2, 128,256,512,1024,2048, 3)
        x_list.reverse()                # len = 5     [B, D, S]  (2, 128,128,64,32,16, 128,256,512,1024,2048)
        x = x_list[0]                   #             [B, D, S]  (2 128 8)
        for i in range(len(self.decode_list)): # 4
            x = self.decode_list[i](xyz_list[i+1], xyz_list[i], x_list[i+1],x)        # > [B, D, S]
            #                       2 256 3   , 2 128 3   , 2 128 256  , 2 512 256      > 2 512 256
            #                       2 512 3   , 2 256 3   , 2 64 512   , 2 256 512      > 2 256 512
            #                       2 1024 3  , 2 512 3   , 2 32 1024  , 2 128 1024     > 2 128 1024
            #                       2 2048 3  , 2 1024 3  , 2 16 2048  , 2 128 2048     > 2 128 2048

        # Global Context
        # x_list    : 2 128 128  -  2 128 256  -  2 64 512  -  2 32 1024  -  2 16 2048
        gmp_list = []
        for i in range(len(x_list)):    # 5
            gmp_list.append(F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1))
            print(x_list[i].shape, self.gmp_map_list[i], self.gmp_map_list[i](x_list[i]).shape, F.adaptive_max_pool1d(self.gmp_map_list[i](x_list[i]), 1).shape)
            print()
            # 1: 2 128 128   MLP(128>64) 2 64 128      pool    2 64 1
            # 2: 2 128 256   MLP(128>64) 2 64 256      pool    2 64 1
            # 3: 2 64 512    MLP(128>64) 2 64 512      pool    2 64 1
            # 4: 2 32 1024   MLP(32>64)  2 64 1024     pool    2 64 1
            # 5: 2 16 2048   MLP(16>64)  2 64 2048     pool    2 64 1
            # gmp_list: 2 64 1, 2 64 1, 2 64 1, 2 64 1, 2 64 1

        global_context = self.gmp_map_end(torch.cat(gmp_list, dim=1)) # 2,320,1  MLP(320>64)  2,64,1

        # cls_token
        cls_token = self.cls_map(cls_label.unsqueeze(dim=-1))  # [b, cls_dim, 1]  2,16 > 2,16,1  MLP(16>64) MLP(64>64)  2,64,1
        x = torch.cat([x, global_context.repeat([1, 1, x.shape[-1]]), cls_token.repeat([1, 1, x.shape[-1]])], dim=1) # 2 256 2048
        # 2 128 2048  -  2 64 2048  -  2 64 2048  >  2 256 2048
        x = self.classifier(x)      # 2 256 2048 MLP(256>128>50) 2 50 2048
        x = F.log_softmax(x, dim=1) # 2 50 2048
        x = x.permute(0, 2, 1)      # 2 2048 50
        return x


def LPTNet(num_classes=50, **kwargs) -> Decoder:
   return Decoder(num_classes=num_classes, points=2048, embed_dim=16, groups=1, res_expansion=1.0,
                activation="relu", bias=True, use_xyz=True, normalize="anchor",
                dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                k_neighbors=[32, 32, 32, 32], reducers=[4, 4, 4, 4], # [2,2,2,2]
                de_dims=[512, 256, 128, 128], de_blocks=[4,4,4,4],
                gmp_dim=64,cls_dim=64, k_attention=[24,24,24,8], **kwargs)


if __name__ == '__main__':

    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 2048) # 2 3 2048
    norm = torch.rand(2, 3, 2048) # 2 3 2048
    cls_label = torch.rand([2, 16]) # 2 16
    print("===> testing modelD ...")
    model_lite = LPTNet()

    print(f'params lite: {trainable_params(model_lite)}')

    out_lite = model_lite(data, norm, cls_label)  # [2,2048,50]
    
    print(out_lite.shape)