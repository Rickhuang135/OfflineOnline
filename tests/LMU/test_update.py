from matplotlib import pyplot as plt
import numpy as np
import torch

from OfflineOnline.LMU.memory_tr import Memory
from OfflineOnline.config.device import DEVICE

# set up variables
test_range = 20
window_size = 20
resolution = 0.01
degree_approx = 10
M = Memory(window_size, 1, degree_approx)
target_functions = [lambda x: x*3+20, np.sin]

# store in memory
for i in range(test_range):
    x_t = [tf(i) for tf in target_functions]
    # M.update(np.array(x_t)) # test_np
    M.update(torch.tensor(x_t, device=DEVICE)) # test_tr

# plot
x = np.arange(-1, 1, resolution/window_size*2)
axis = np.arange(0, window_size, resolution)
result_y = M.reconstruct(x)
for target_function, result_graph in zip(target_functions, result_y):
    plt.plot(axis, target_function(np.flip(np.arange(test_range-window_size, test_range, resolution))), label = "target")
    plt.plot(axis, result_graph, label = "memory reconstruction")
    plt.legend()
    plt.show()