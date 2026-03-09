import torch

from random import random

def choose_action(action_values: torch.Tensor, epsilon: float) -> torch.Tensor:
    na = action_values.shape[-1]
    action_values = action_values.reshape(-1, na)
    best_actions = torch.argmax(action_values, dim=-1)
    rand = torch.rand(action_values.shape[0])
    random_actions = torch.randint_like(best_actions,0,na)
    return torch.where(rand<epsilon, random_actions, best_actions).squeeze()