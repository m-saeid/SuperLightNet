try:
    from models.SLNet import *
except:
    from models.SLNet import *



def SLNet_edge0_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["fps", 1.0]], **kwargs)

def SLNet_edge25_k3_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.03 , 0.25], ["fps", 0.75]], **kwargs)


def SLNet_edge25_k5_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.05 , 0.25], ["fps", 0.75]], **kwargs)


def SLNet_edge25_k25_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.25 , 0.25], ["fps", 0.75]], **kwargs)


def SLNet_edge25_k50_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.5 , 0.25], ["fps", 0.75]], **kwargs)


def SLNet_edge25_k75_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.75 , 0.25], ["fps", 0.75]], **kwargs)


def SLNet_edge25_k100_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 1.0 , 0.25], ["fps", 0.75]], **kwargs)


def SLNet_edge50_k3_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.03 , 0.5], ["fps", 0.5]], **kwargs)


def SLNet_edge50_k5_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.05 , 0.5], ["fps", 0.5]], **kwargs)


def SLNet_edge50_k25_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.25 , 0.5], ["fps", 0.5]], **kwargs)


def SLNet_edge50_k50_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.5 , 0.5], ["fps", 0.5]], **kwargs)


def SLNet_edge50_k75_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.75 , 0.5], ["fps", 0.5]], **kwargs)


def SLNet_edge50_k100_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 1.0 , 0.5], ["fps", 0.5]], **kwargs)


def SLNet_edge75_k3_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.03 , 0.75], ["fps", 0.25]], **kwargs)


def SLNet_edge75_k5_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.05 , 0.75], ["fps", 0.25]], **kwargs)


def SLNet_edge75_k25_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.25 , 0.75], ["fps", 0.25]], **kwargs)


def SLNet_edge75_k50_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.5 , 0.75], ["fps", 0.25]], **kwargs)


def SLNet_edge75_k75_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.75 , 0.75], ["fps", 0.25]], **kwargs)


def SLNet_edge75_k100_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 1.0 , 0.75], ["fps", 0.25]], **kwargs)


def SLNet_edge100_k3_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.03 , 1.0]], **kwargs)


def SLNet_edge100_k5_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.05 , 1.0]], **kwargs)


def SLNet_edge100_k25_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.25 , 1.0]], **kwargs)


def SLNet_edge100_k50_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.5 , 1.0]], **kwargs)


def SLNet_edge100_k75_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 0.75 , 1.0]], **kwargs)


def SLNet_edge100_k100_fps(num_classes=40, **kwargs) -> Model:
    return Model(points=1024, class_num=num_classes, embed_dim=16, groups=1, res_expansion=0.25,
                    activation="relu", bias=False, use_xyz=True, normalize="anchor",                    # use_xyz=True > False
                    dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                    k_neighbors=[24,24,24,24], reducers=[2, 2, 2, 2],
                    details=[["edge", 1.0 , 1.0]], **kwargs)




if __name__ == '__main__':
    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    #model = pointMLP()
    model = SLNet_edge100_k75_fps()
    print(f'number of params newModel: {trainable_params(model)}')
    
    out = model(data)
    print(out.shape)