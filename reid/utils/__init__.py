from __future__ import absolute_import

import torch


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def cal_dist(qFeat, gFeat):
    m, n = qFeat.size(0), gFeat.size(0)
    x = qFeat.view(m, -1)
    y = gFeat.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
             torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m


def split_cam(camCount, mRatio):
    return torch.randperm(camCount)[:int(mRatio * camCount)]


# generate new dataset and calculate cluster centers
def generate_pseudo_labels(cluster_id, inputFeat):
    with_id, witho_id = inputFeat[cluster_id != -1], inputFeat[cluster_id == -1]
    disMat = cal_dist(with_id, witho_id)
    # relabel images
    neighbour = disMat.argmin(0).cpu().numpy()
    newID = cluster_id[cluster_id != -1][neighbour]
    cluster_id[cluster_id == -1] = newID
    return torch.from_numpy(cluster_id).long()
