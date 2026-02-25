from matplotlib import pyplot as plt
import numpy as np

from OfflineOnline.LMU.memory_tr import Memory

# set up variables
test_range = 20
window_size = 20
resolution = 0.01
dimensions = 10
M = Memory(window_size, dimensions)
# target_function = lambda x: x*3+20
target_function = np.sin

# store in memory
for i in range(test_range):
    M.update(target_function(i))
result_function = M.reconstruct()

# plot
x = np.arange(-1, 1, resolution/window_size*2)
axis = np.arange(0, window_size, resolution)
plt.plot(axis, target_function(np.flip(np.arange(test_range-window_size, test_range, resolution))), label = "target")
plt.plot(axis, result_function(x), label = "memory reconstruction")
plt.legend()
plt.show()