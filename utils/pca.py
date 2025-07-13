import torch

def pca(x, dims=2):
    # x: (N, D)

    x = x - x.mean(dim=1, keepdim=True) # (N, D)
    _, S, Vt = torch.linalg.svd(x) # (D,) (D, D)
    return (torch.diag(S)@Vt)[:dims,:], S[:dims] # (dims, D), (dims,)
