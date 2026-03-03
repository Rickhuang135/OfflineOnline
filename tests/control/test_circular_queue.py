import torch
from OfflineOnline.control.circular_queue import CircularQueue

window_length = 5
item_dimension = 3
A = CircularQueue(window_length, item_dimension, device='cpu') # type:ignore

for i in range(0, window_length*item_dimension*2, item_dimension):
    print(A.forward_view().T)
    new_item = torch.arange(i, item_dimension+i)
    A.append(new_item)