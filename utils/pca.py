import torch

def pca(x, dims=2):
    # x: (D, N)

    x = x - x.mean(dim=1, keepdim=True) # (D, N)
    _, S, Vt = torch.linalg.svd(x) # (N,) (N, N)
    return (torch.diag(S)@Vt)[:dims,:], S[:dims] # (dims, N), (dims,)
