import torch

from OfflineOnline.control.epsilon_greedy import choose_action

A = torch.arange(10)
# .reshape(2,5)

print(choose_action(A, 0))
print(choose_action(A, 0.5))
print(choose_action(A, 1))