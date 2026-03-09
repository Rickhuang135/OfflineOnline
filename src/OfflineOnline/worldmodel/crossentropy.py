import torch

from OfflineOnline.config.latent import split

class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(
            self, 
            true_dist: torch.Tensor, 
            predicted_dist: torch.Tensor,
            ):
        meantd, vartd = split(true_dist)
        meanpd, varpd = split(predicted_dist)
        term1 = torch.log(varpd)
        term2num = vartd**2 + (meantd - meanpd)**2
        term2den = 2 * varpd**2

        return (term1 + term2num/term2den).sum(dim=-1)