import torch
from torch import nn

from OfflineOnline.config.device import DEVICE

class Reconstruct(nn.Module):
    def __init__(
            self, 
            nl: int, # size of latent state
            n_observation: int, # size of observations
            device = DEVICE,
            ):
        self.l0 = nn.Linear(nl, n_observation//4)
        self.a0 = nn.ReLU()
        self.l1 = nn.Linear(n_observation//4, n_observation//2)
        self.a1 = nn.ReLU()
        self.l_observation = nn.Linear(n_observation//2, n_observation)
        self.a_observation = nn.ReLU()
        self.reward_and_continuation = nn.Linear(n_observation//2, 2) # one dimension for reward and continuation each
        self.to(device)

    def forward(self, latent_state: torch.Tensor):
        x = self.a0(self.l0(latent_state))
        x = self.a1(self.l1(x))
        observation = self.a_observation(self.l_observation(x))
        reward_and_continuation = self.reward_and_continuation(x)
        return observation, reward_and_continuation