import torch
from OfflineOnline.control import SarsaLambda, Qlearning
torch.set_printoptions(sci_mode= False, precision=2)

nx = 4
window_size = 4
gamma = 1
n_parallel = 3
na = n_parallel
steps = 200

control = SarsaLambda(nx, na, window_size, gamma, n_parallel, device="cpu") # type:ignore

x = torch.arange(nx*n_parallel, dtype=torch.float32).reshape(n_parallel,nx)

print(x)

control.get_action(x, 0.5)
control.get_action(x*-1, 0.5)
control.backprop()
# states = control.past_states.forward_view()
# states_flat = states.reshape((window_size+1)*n_parallel, nx)
# pred_values = control.Q(states_flat)
# print(states)
# print(states_flat)
# print(pred_values)