import torch
from torch import nn

from OfflineOnline.environment.device import DEVICE

class FlatLinear(nn.Linear):
    def __init__(self, nr: int, nc: int, no:int, device = DEVICE, dtype = None):
        super().__init__(nr * nc, no, device=device, dtype = dtype)
        self.number_input_rows = nr
        self.number_input_columns = nc
        if no == nc: # initialize 1 to 1 match if ouputs = columns
            A = torch.diag(torch.ones(nc, device=device))
            A = A.repeat(1, nr) / nr
            with torch.no_grad():
                self.weight.copy_(A)
                self.bias.zero_()

    def forward(self, input: torch.Tensor):
        x = torch.squeeze(input.reshape(-1, self.number_input_rows * self.number_input_columns))
        y = super().forward(x)
        return y
        