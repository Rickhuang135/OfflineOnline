import torch

from OfflineOnline.config.device import DEVICE
from .action_value import ActionValue
from .circular_queue import CircularQueue
from .epsilon_greedy import choose_action

class SarsaLambda:
    def __init__(
        self,
        inputs: int,
        na: int, # number of discrete actions to choose from
        window_size: int,
        gamma: float,
        lambDa: float | None = None,
        device = DEVICE,
    ):
        self.inputs = inputs
        self.na = na
        self.window_size = window_size
        self.Q = ActionValue(inputs, na)
        self.past_values = CircularQueue(window_size, 1)
        self.gamma = gamma
        if lambDa is None:
            self.lambDa = 0.001 ** (1/window_size) # where 0.001 is some small number
        else:
            self.lambDa = lambDa
        self.elligibility = (self.lambDa*gamma) ** torch.arange(window_size-1, -1, -1, device=device)
        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.Q.parameters())
    
    def get_action(self, state: torch.Tensor, epsilon: float, write_to_window = True):
        action_values = self.Q(state)
        action_ind = choose_action(action_values, epsilon)
        if write_to_window:
            self.past_values.append(action_values[action_ind])
        return action_ind
    
    def backprop(self, td: torch.Tensor):
        pred_values = self.past_values.forward_view()
        targ_values = pred_values.detach() + self.gamma*td*self.elligibility
        loss = self.criterion(targ_values, pred_values)
        loss.back(retain_graph = True)
        self.optimiser.step()
        return loss
        
