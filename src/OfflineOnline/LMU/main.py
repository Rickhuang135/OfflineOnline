import torch
from torch.nn import Linear

from OfflineOnline.config.device import DEVICE
from .flattened_layer import FlatLinear
from .memory_tr import Memory

class Layer(torch.nn.Module):
    def __init__(
            self, 
            nx: int, # dimension of inputs 
            nh: int, # dimension of hidden states
            nu: int, # dimension of memory states
            nt: int, # number of time steps to remember
            na: int = 100, # degree of approximation in memory states
            device = DEVICE,
            dtype = torch.float32
            ):
        super().__init__()
        # encoder
        self.ex = Linear(nx, nu, device=device, dtype=dtype)
        self.eh = Linear(nh, nu, device=device, dtype=dtype)
        self.em = FlatLinear(na, nu, nu, device=device, dtype=dtype)
        self.A1 = torch.nn.Tanh()

        # create values
        self.memory = Memory(nt, nu, na, device=device, dtype=dtype)
        self.h_t = torch.zeros(nh, device=device, dtype=dtype)
        
        # non-linear
        self.Wx = Linear(nx, nh, device=device, dtype=dtype)
        self.Wh = Linear(nh, nh, device=device, dtype=dtype)
        self.Wm = FlatLinear(na, nu, nh, device=device, dtype=dtype)
        self.A2 = torch.nn.LeakyReLU()

    
    def forward(self, x):
        u_t = self.A1(self.ex(x) + self.eh(self.h_t) + self.em(self.memory.values))
        self.memory.update(u_t)
        h_t1 = self.A2(self.Wx(x) + self.Wh(self.h_t) + self.Wm(self.memory.values))
        self.h_t = h_t1
        return h_t1

    