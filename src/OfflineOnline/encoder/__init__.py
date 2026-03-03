import torch
from torch import nn

from OfflineOnline.LMU.main import Layer as LMULayer
from OfflineOnline.config.device import DEVICE
from .gaussify import Gaussify

class Encoder(nn.Module):
    def __init__(
            self,
            nx: int,# dimension of each (flattened) input
            nl: int, # dimension of latent state
            window_size: int = 10,
            hidden_size: int = 5,
            na: int = 1, # dimension of actions
            device = DEVICE,
            ):
        total_inputs = nx + na + 2 # reward and continuation always has 1 dimension each
        self.total_inputs = total_inputs
        self.LMU = LMULayer(total_inputs, hidden_size, hidden_size, window_size, max(window_size*2//3, 2))
        self.gaussify = Gaussify(hidden_size, nl)
        self.device = device
        self.to(device)

    def forward(self, x: torch.Tensor, reward_and_continuation: torch.Tensor, last_action: torch.Tensor):
        x_all = torch.concat([x, reward_and_continuation, last_action], dim=-1)
        hidden_state = self.LMU(x_all)
        means, stds = self.gaussify(hidden_state)
        return means, stds
