import numpy as np
import torch

from OfflineOnline.environment.device import DEVICE
from .memory_np import Memory as Memory_np
from .generate_polynomials import phi as p

class Memory(Memory_np):
    def __init__(self, window_size: int, number_states: int = 1, degree_approx: int = 20, device = DEVICE, dtype = torch.float64):
        super().__init__(window_size=window_size, number_states=number_states, degree_approx=degree_approx)
        self.A = torch.tensor(self.A, device = device, dtype = dtype)
        self.B = torch.tensor(self.B, device = device, dtype = dtype)
        self.values = torch.zeros((degree_approx, number_states), device=device, dtype = dtype)
    

    def reconstruct(self, x: np.ndarray):
        stack = p(x, self.degree_approx)
        return (stack.T@(self.values.cpu().numpy())).T

