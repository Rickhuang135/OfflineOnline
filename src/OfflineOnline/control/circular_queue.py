import torch

from OfflineOnline.config.device import DEVICE

class CircularQueue:
    def __init__(self, length, item_dimension, device = DEVICE):
        self.queue = torch.zeros((length, item_dimension), device=device)
        self.write_ind = 0
        self.end_ind = 0
        self.length = length
        self.item_dimension = item_dimension

    def __getitem__(self, index):
        return self.queue[(self.write_ind+index)%self.length, :]
    
    def append(self, item: torch.Tensor):
        self.queue[self.write_ind, :] = item
        self.write_ind = (self.write_ind + 1) % self.length

    def __repr__(self):
        return str(self.queue)
    
    def forward_view(self):
        behind_write = self.queue[0: self.write_ind, :]
        infront_write = self.queue[self.write_ind:, :]
        return torch.concat((infront_write, behind_write), dim = 0)