import torch
from OfflineOnline.LMU.main import Layer

nx = 40
nh = 5
nu = 10
nt = 500
na = 100

LMU = Layer(nx, nh, nu, nt, na, 'cpu') # type:ignore

r = LMU(torch.rand(nx))
print(r)