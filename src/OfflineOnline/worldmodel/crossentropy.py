import torch

class CrossEntropyLoss:
    def __init__(self):
        pass

    def __call__(
            self, 
            meantd: torch.Tensor, # mean of true distribution
            vartd: torch.Tensor, # variance of true distribution
            meanpd: torch.Tensor, # mean of predicted distribution
            varpd: torch.Tensor, # variance of predicted distribution
            ):
        term1 = torch.log(varpd)
        term2num = vartd**2 + (meantd - meanpd)**2
        term2den = 2 * varpd**2

        return (term1 + term2num/term2den).sum(dim=-1)