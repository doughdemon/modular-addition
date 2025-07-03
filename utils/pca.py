import torch

def pca(x, dims=2):
    A = x.transpose(0,1)

    A = A - A.mean(dim=0)
    U, S, _ = torch.linalg.svd(A)
    return (U@torch.diag(S))[:, :dims], S[:dims]
