import torch
from torch import nn

from OfflineOnline.config.latent import combine
from OfflineOnline.config.device import DEVICE

class DynamicsModel(nn.Module):
    def __init__(
            self, 
            nl: int, # dimension of latent states
            na: int = 1, # dimension of action
            device = DEVICE,
            ):
        super().__init__()
        self.nl = nl
        self.na = na
        self.l1 = nn.Linear(nl + na, nl*4) # extra input for action
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(nl*4, nl*2)
        self.a2 = nn.ReLU()
        self.l_means = nn.Linear(nl*2, nl)
        self.a_means = nn.ReLU()
        self.l_stds = nn.Linear(nl*2, nl)
        self.a_stds = nn.Softplus() # must have non-zero positive output
        self.l_reward_and_continuation = nn.Linear(nl*2, 2)
        self.device = device
        self.to(device)

    def forward(self, latent: torch.Tensor, action: torch.Tensor):
        x = torch.concat([latent, action], dim=-1)
        x = self.a1(self.l1())
        x = self.a2(self.l2(x))
        means = self.a_means(self.l_means(x))
        stds = self.a_stds(self.l_stds(x))
        reward_and_continuation = self.l_reward_and_continuation(x)
        return combine(means,stds), reward_and_continuation