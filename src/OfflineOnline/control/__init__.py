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
        n_parallel: int,
        lambDa: float | None = None,
        device = DEVICE,
    ):
        self.inputs = inputs
        self.n_parallel = n_parallel
        self.na = na
        self.window_size = window_size
        self.filled_window_size = 0
        self.no_reward = self.no_end = torch.zeros(n_parallel, device=device)

        self.Q = ActionValue(inputs, na, device = device)
        self.past_states = CircularQueue(window_size + 1, n_parallel * inputs, device = device) # Store states for re-inference
        self.past_actions = CircularQueue(window_size + 1, n_parallel, device=device, dtype = torch.int32) # Store actions as indicies for re-inference
        self.gamma = gamma
        if lambDa is None:
            self.lambDa = 0.001 ** (1/window_size) # where 0.001 is some small number
        else:
            self.lambDa = lambDa
        self.elligibility = ((self.lambDa*gamma) ** torch.arange(window_size-1, -1, -1, device=device)).unsqueeze(1)
        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.Q.parameters(), lr=0.05)
    
    def get_action(self, state: torch.Tensor, epsilon: float, write_to_window = True):
        with torch.no_grad():
            action_values = self.Q(state).reshape(self.n_parallel, self.na)
            action_ind = choose_action(action_values, epsilon) # array of length n_parallel
            if write_to_window:
                self.write_to_window(state, action_ind)
            return action_ind
    
    def write_to_window(self, states: torch.Tensor, action_ind: torch.Tensor):
        self.past_states.append(states.flatten())
        self.past_actions.append(action_ind)
        if self.filled_window_size <= self.window_size:
            self.filled_window_size += 1

    def process_past_values(self, all_pred_values, actions):
        pred_values = torch.gather(all_pred_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1) # shape ( window_size + 1, n_parallel )
        return pred_values, pred_values[-1]
    
    def backprop(self, reward: torch.Tensor | None = None, end: torch.Tensor | None = None):
        if reward is None:
            reward = self.no_reward
        if end is None:
            end = self.no_end
        
        length = self.filled_window_size
        actions = self.past_actions.forward_view()[-length:] # shape ( window_size + 1, n_parallel )
        states_flat = self.past_states.forward_view()[-length:].reshape((length)*self.n_parallel, self.inputs)
        self.Q.train()
        self.optimiser.zero_grad()
        inference_outcome = self.Q(states_flat)
        all_pred_values = inference_outcome.reshape(length, self.n_parallel, self.na)
        pred_values, latest_value = self.process_past_values(all_pred_values, actions)
        td = reward + (latest_value * self.gamma) * (1-end) - pred_values[-2]

        trainable_values = pred_values[:-1] # do not include last value for training
        targ_values = trainable_values.detach() + td.detach()*self.elligibility[1-length:]
        print("\n\n")
        # print(td.detach()*self.elligibility[1-length:])
        # print(f"actions \n{actions}")
        print(f"states \n{states_flat[0]}\n")
        print(f"inference out come \n{inference_outcome[0]}\n")
        # print(f"all_pred_values \n{all_pred_values}")
        # print(f"pred_values \n{pred_values}")
        print(f"actions \n{actions[0]}\n")
        # print(f"td \n{td}")
        print(f"targ_values \n{targ_values}\n")
        print(f"trainable_values \n{trainable_values}\n")
        loss = self.criterion(targ_values, trainable_values)
        loss.backward()
        self.optimiser.step()
        return loss
        
    def drop_traces(self):
        self.past_states.reset()
        self.past_actions.reset()
        self.filled_window_size = 0

class Qlearning(SarsaLambda):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def process_past_values(self, all_pred_values, actions):
        all_last_values = all_pred_values[-1]
        best_values = torch.max(all_last_values, dim=-1).values
        pred_values = torch.gather(all_pred_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1) # shape ( window_size + 1, n_parallel )
        return pred_values, best_values