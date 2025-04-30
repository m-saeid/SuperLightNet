import os
import glob
import h5py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ── Helpers for positional encoding ────────────────────────────────────────────
def fourier_encode(xyz, scales):
    """
    xyz: [1, N, 3]
    returns [1, N, 2*len(scales)*3]
    """
    feats = []
    for s in scales:
        for fn in (torch.sin, torch.cos):
            feats.append(fn(xyz * (2**s * np.pi)))
    return torch.cat(feats, dim=-1)

def gaussian_rbf_encode(xyz, knn=16, sigma=0.1):
    """
    xyz: [1, N, 3]
    returns [1, N, knn]
    """
    # pairwise distance
    dists = torch.cdist(xyz, xyz)              # [1, N, N]
    knn_d, _ = torch.topk(dists, knn, dim=-1, largest=False)
    return torch.exp(-knn_d**2 / (sigma**2))

def load_data(partition):
    PATH = [Path("/home/saeid/Desktop/saeid/datasets/"),
            Path("/home/anil/Desktop/saeid/datasets/"),
            Path("/home/fovea/Desktop/Datasets_3080/ModelNet40/"),
            Path("/home/iris/Desktop/Datasets_2080/ModelNet40/"),
            Path("/home/zeus/Desktop/Datasets_1080/ModelNet40/")]
    if PATH[0].exists():
        DATA_DIR = PATH[0]
    elif PATH[1].exists():
        DATA_DIR = PATH[1]
    elif PATH[2].exists():
        DATA_DIR = PATH[2]
    elif PATH[3].exists():
        DATA_DIR = PATH[3]
    elif PATH[4].exists():
        DATA_DIR = PATH[4]
    else:
        raise ValueError(f"Dataset not found in {PATH}")
    
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label
def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train',
                 use_hybrid_pe=False, pe_scales=(0,1,2,4,8,16), pe_sigma=0.1):
        self.data, self.label = load_data(partition)
        self.num_points   = num_points
        self.partition    = partition
        self.use_hybrid_pe = use_hybrid_pe
        self.pe_scales    = pe_scales
        self.pe_sigma     = pe_sigma


    def __getitem__(self, idx):
        # 1) Base pointcloud [N,3]
        pc = self.data[idx][:self.num_points]            # (N,3)
        label = int(self.label[idx])

        # 2) Train-time augmentation
        if self.partition == 'train':
            pc = translate_pointcloud(pc)
            np.random.shuffle(pc)

        # 3) Positional encoding?
        if self.use_hybrid_pe:
            # to tensor [1,N,3]
            xyz = torch.from_numpy(pc).unsqueeze(0)      # (1,N,3)
            fourier = fourier_encode(xyz, self.pe_scales)    # (1,N, 2*len*3)
            gauss   = gaussian_rbf_encode(xyz, knn=16, sigma=self.pe_sigma)  # (1,N,16)
            enriched = torch.cat([xyz, fourier, gauss], dim=-1)  # (1,N,3+...)
            pc_out = enriched.squeeze(0)                 # (N, D_enriched)
        else:
            pc_out = torch.from_numpy(pc)                # (N,3)

        return pc_out, label

    def __len__(self):
        return self.data.shape[0]


# ── test loader snippet ────────────────────────────────────────────────────────
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    ds = ModelNet40(1024, 'train', use_hybrid_pe=True)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    for pts, lbl in loader:
        print("pts:", pts.shape, "lbl:", lbl.shape)
        break