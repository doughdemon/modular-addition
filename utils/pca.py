import torch

def pca(x, dims=2):
    A = x.transpose(0,1)

    if True:
        torch.manual_seed(1)
        V = torch.pca_lowrank(A)[dims]
        return torch.matmul(A, V[:, :dims])
    else:
        A = A - A.mean(dim=0)
        U, S, _ = torch.linalg.svd(A)
        return (U@torch.diag(S))[:, :dims]
