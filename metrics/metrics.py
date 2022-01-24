import torch
from tqdm import tqdm

# Import CUDA version of approximate EMD, from https://github.com/zekunhao1995/pcgan-pytorch/
from metrics.StructuralLosses.match_cost import match_cost
from metrics.StructuralLosses.nn_distance import nn_distance


def compute_cd(x, y, reduce_func=torch.mean):
    d1, d2 = nn_distance(x, y)
    return reduce_func(d1, dim=1) + reduce_func(d2, dim=1)


def compute_emd(x, y):
    return match_cost(x, y) / x.size(1)


def compute_pairwise_cd_emd(x, y, batch_size=32):
    NX, NY, cd, emd = x.size(0), y.size(0), [], []
    y = y.contiguous()
    for i in tqdm(range(NX)):
        cdx, emdx, xi = [], [], x[i]
        for j in range(0, NY, batch_size):
            yb = y[j : j + batch_size]
            xb = xi.view(1, -1, 3).expand_as(yb).contiguous()
            cdx.append(compute_cd(xb, yb).view(1, -1))
            emdx.append(compute_emd(xb, yb).view(1, -1))
        cd.append(torch.cat(cdx, dim=1))
        emd.append(torch.cat(emdx, dim=1))
    cd, emd = torch.cat(cd, dim=0), torch.cat(emd, dim=0)
    return cd, emd


def compute_mmd_cov(dxy):
    _, min_idx = dxy.min(dim=1)
    min_val, _ = dxy.min(dim=0)
    mmd = min_val.mean()
    cov = min_idx.unique().numel() / dxy.size(1)
    cov = torch.tensor(cov).to(dxy)
    return mmd, cov


@torch.no_grad()
def compute_metrics(x, y, batch_size):
    cd_yx, emd_yx = compute_pairwise_cd_emd(y, x, batch_size)
    mmd_cd, cov_cd = compute_mmd_cov(cd_yx.t())
    mmd_emd, cov_emd = compute_mmd_cov(emd_yx.t())
    return {
        "COV-CD": cov_cd.cpu(),
        "COV-EMD": cov_emd.cpu(),
        "MMD-CD": mmd_cd.cpu(),
        "MMD-EMD": mmd_emd.cpu(),
    }
