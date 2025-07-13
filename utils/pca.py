import torch

def pca(x, dims=2):
    # x: (N, D)
    A = x.transpose(0,1) # (D, N)

    A = A - A.mean(dim=0) # (D, N)
    U, S, _ = torch.linalg.svd(A) # (D, D) (D,)
    return (U@torch.diag(S))[:, :dims], S[:dims] # (D, dims), (dims,)
