import torch
from torch import nn

from OfflineOnline.config.latent import combine
from OfflineOnline.config.device import DEVICE

class Gaussify(nn.Module):
    def __init__(
            self,
            n_input: int,
            nl: int, # dimension of latent space
            device = DEVICE,
    ):
        self.l_means = nn.Linear(n_input, nl)
        self.a_means = nn.ReLU()
        self.l_stds = nn.Linear(n_input, nl)
        self.a_stds = nn.Softplus() # must have non-zero positive output
        self.to(device)

    def forward(self, x):
        means = self.a_means(self.l_means(x))
        stds = self.a_stds(self.l_stds(x))
        return combine(means, stds)