import torch
from torch import nn
torch.set_printoptions(sci_mode= False)

s1 = torch.tensor([0,0], dtype=torch.float32)
s2 = torch.tensor([1,1], dtype=torch.float32)
r_correct = torch.tensor(0, dtype = torch.float32)
r_wrong = torch.tensor(-5, dtype= torch.float32)

t1 = torch.tensor([0,-5], dtype=torch.float32)
t2 = torch.tensor([-5,0], dtype=torch.float32)

class Xpnn(torch.nn.Module):
    def __init__(self, ni, no):
        super(Xpnn, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(ni, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, no)
        )
    def forward(self, x):
        return self.encoder(x)

steps = 25
xp = Xpnn(2,2)
criterion = torch.nn.MSELoss()
optimiser = torch.optim.Adam(xp.parameters(), lr=0.05)
historical_loss = []
for i in range(steps):
    optimiser.zero_grad()
    if i%2 == 0:
        state = s1
        target = t1
    else:
        state = s2
        target = t2
    output = xp(state)
    loss = criterion(target, output)
    loss.backward()
    optimiser.step()
    historical_loss.append(loss.detach())
print(torch.stack(historical_loss))
print(xp(s1))
print(xp(s2))