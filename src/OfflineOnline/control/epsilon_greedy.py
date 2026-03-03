import torch

from random import random

def choose_action(action_values: torch.Tensor, epsilon: float) -> torch.Tensor:
    if random() < epsilon: # make random action
        if len(action_values.shape) == 2:
            size = (action_values.shape[0],)
        else:
            size = (1,)
        return torch.randint(0, action_values.shape[-1], size, device=action_values.device)
    return torch.argmax(action_values, dim=-1)