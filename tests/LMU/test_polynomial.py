from OfflineOnline.LMU.generate_polynomials import phi as p
from matplotlib import pyplot as plt

import numpy as np

dimensions = 4

x = np.arange(-1,1, 0.01)

stack = p(x, dimensions)

for function in stack:
    plt.plot(x,function)
plt.show()