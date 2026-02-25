import numpy as np
import torch

from OfflineOnline.environment.device import DEVICE
from .memory_np import Memory as Memory_np
from .generate_polynomials import phi as p

class Memory(Memory_np):
    def __init__(self, window_size: int, dimensions: int = 20, device = DEVICE, dtype = torch.float64):
        super().__init__(window_size, dimensions)
        self.A = torch.tensor(self.A, device = device, dtype = dtype)
        self.B = torch.tensor(self.B, device = device, dtype = dtype)
        self.values = torch.zeros(dimensions, device=device, dtype = dtype)

    def reconstruct(self):
        def approx(x: np.ndarray): # expects values betwee -1 and 1
            stack = p(x, self.dimensions)
            return (stack.T@(self.values.cpu().numpy()))
        return approx