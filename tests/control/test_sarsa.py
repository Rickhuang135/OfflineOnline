import torch
from OfflineOnline.control import SarsaLambda, Qlearning
torch.set_printoptions(sci_mode= False)

nx = 2
window_size = 4
gamma = 1
n_parallel = 3
na = n_parallel
steps = 400

control = SarsaLambda(nx, na, window_size, gamma, n_parallel, device="cpu") # type:ignore
# control = Qlearning(nx, na, window_size, gamma, n_parallel, device="cpu") # type:ignore
if n_parallel == 1:
    x = torch.arange(nx, dtype = torch.float32)
else:
    x = torch.arange(nx*n_parallel, dtype=torch.float32).reshape(n_parallel,nx)
    # x = torch.arange(n_parallel, dtype= torch.float32).unsqueeze(1).repeat(1,nx)
    # x = torch.Tensor(
    #     [[1,0],[0,1]]
    # )
print(x)

# train the network to always output the index of the parallel process
correct_actions = torch.arange(n_parallel)
print(correct_actions)
r_correct = 0
r_wrong = -5
historic_loss = []
end = torch.ones(n_parallel)
a0 = control.get_action(x, 1)
# print(control.past_values.forward_view())
for i in range(steps):
    actions = control.get_action(x, 1-i/steps)
    # print(control.past_values.forward_view())
    rewards = torch.where(a0==correct_actions, r_correct, r_wrong)
    print(a0)
    print(rewards)
    # print(actions)
    # print(rewards)
    a0 = actions
    historic_loss.append(control.backprop(rewards, end=end).detach())
print(torch.stack(historic_loss))
print(x)
print(control.Q(x))
print(control.get_action(x, 0))
