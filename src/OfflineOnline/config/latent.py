import torch

def split(latent: torch.Tensor):
    size = latent.shape[-1]
    latent = latent.reshape(-1, size)
    half = size//2
    means = latent[:, :half]
    vars = latent[:, half:]
    return means, vars

def combine(means: torch.Tensor, varience: torch.Tensor):
    return torch.concat([means, varience], dim = -1)
