import torch
import torch.nn as nn


def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def remove_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: indices of the points to remove, [B, S]
    Return:
        new_points: points data with specified indices removed, [B, N-S, C]
    """
    B, N, C = points.shape
    device = points.device
    
    # Create a mask for each batch
    mask = torch.ones((B, N), dtype=torch.bool).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(-1, 1)
    mask[batch_indices, idx] = False

    # Apply mask to remove points and reshape
    new_points = points[mask].view(B, -1, C)
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def edgeSampling(xyz, k, s):
    """
    Input:
        s: sampling numbers
        xyz: all points, [B, N, C]
        k = number of neighbors to attention
    Return:
        samples_idx: sampled pointcloud index, [B, npoint]
    """
    number_of_k = int(xyz.shape[1] * k) # int(s * k)
    sqrdists = square_distance(xyz, xyz)    # 2,1024,1024
    sqrdists = -1*((-1*sqrdists).topk(number_of_k)[0])         # 2,1024,number_ok_k
    distances = sqrdists.sum(dim=-1)        # 2,1024
    idx = distances.topk(s, dim=-1)[1]      # 2,512
    return idx


def sampling(xyz, points, details, S):
    """
    Perform sampling based on details.
    
    Args:
        xyz: Input points, tensor of shape [B, N, 3]
        points: Additional points data, tensor of shape [B, N, d]
        details: List of sampling methods and their parameters
        S: Total number of samples to obtain

    Returns:
        new_xyz: Sampled xyz points, tensor of shape [B, S, 3]
        new_points: Sampled points data, tensor of shape [B, S, d]
    """
    N = xyz.shape[1]
    s = [int(d[-1] * S) for d in details]
    if sum(s) != S:
        diff = S - sum(s)
        s[-1] += diff

    new_xyz = []
    new_points = []
    
    for i in range(len(details)):
        if details[i][0] == "edge":
            idx = edgeSampling(xyz, details[i][1], s[i])
        elif details[i][0] == "fps":
            idx = farthest_point_sample(xyz, s[i]).long()
        else:
            raise ValueError(f"Unknown sampling method: {details[i][0]}")

        new_xyz.append(index_points(xyz, idx))
        new_points.append(index_points(points, idx))

        xyz = remove_points(xyz, idx)
        points = remove_points(points, idx)

    new_xyz = torch.cat(new_xyz, dim=1)
    new_points = torch.cat(new_points, dim=1)

    return new_xyz, new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx
