import torch

from OfflineOnline.control import SarsaLambda, Qlearning
torch.set_printoptions(sci_mode= False)

from random import random

s1 = torch.tensor([0,0], dtype=torch.float32)
s2 = torch.tensor([1,1], dtype=torch.float32)
r_correct = torch.tensor(0, dtype = torch.float32)
r_wrong = torch.tensor(-5, dtype= torch.float32)

def get_reward(state, action):
    if torch.all(state==s1) and action == 0:
        return r_correct
    elif torch.all(state == s2) and action == 1:
        return r_correct
    else:
        return r_wrong
    
control = SarsaLambda(2, 2, 1, 0.8, 1, device="cpu") # type:ignore
end = torch.ones(1)
steps = 50
historic_loss =[]
for i in range(steps):
    if i%2 == 0:
        state = s1
    else:
        state = s2
    control.drop_traces()
    action = control.get_action(state, 1)
    # action = control.get_action(state, 1-i/20)
    reward = get_reward(state, action)
    action = control.get_action(state, 1-i/20) # doesn't matter
    print(f"end \n{end}")
    historic_loss.append(control.backprop(reward, end).detach())
print(torch.stack(historic_loss))
print(control.Q(s1))
print(control.Q(s2)) # the actions aren't filtering though properly