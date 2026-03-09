import torch
from torch import nn

from OfflineOnline.config.device import DEVICE

class ActionValue(nn.Module):
    def __init__(
            self, 
            n_latent: int, # dimension of latents
            na: int, # number of discrete actions
            device = DEVICE,
            ):
        super(ActionValue, self).__init__()
        self.l0 = nn.Linear(n_latent, n_latent*2)
        self.a0 = nn.ReLU()
        self.l1 = nn.Linear(n_latent*2, na*2)
        self.a1 = nn.ReLU()
        self.lo = nn.Linear(na*2, na)
        self.to(device)
    
    def forward(self, latent):
        x = self.a0(self.l0(latent))
        x = self.a1(self.l1(x))
        return self.lo(x)

