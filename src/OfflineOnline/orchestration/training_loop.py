import torch
import torch.multiprocessing as mp

from OfflineOnline.config.hyperparams import n_parallel

from OfflineOnline.encoder import Encoder
from OfflineOnline.encoder.reconstruct import Reconstruct
from OfflineOnline.worldmodel import DynamicsModel
from OfflineOnline.worldmodel import CrossEntropyLoss
from OfflineOnline.control import SarsaLambda


async def train(
        encoder: Encoder,
        reconstructor: Reconstruct,
        dynamics: DynamicsModel,
        control: SarsaLambda,
        action_buffer: mp.Queue,
        state_buffer: mp.Queue,
        steps: int,
        batch_size: int = n_parallel,
):
    device = dynamics.device
    epsilon = 1
    gamma = control.gamma
    # world model variables
    a0 = torch.zeros(batch_size, device=device)
    l0 = None
    v0 = None
    lpred = None
    RnC0 = None
    RnCpred = None
    # control traning variables
    ar = None
    lr0 = None
    lr1 = None
    vr = None
    RnCr0 = None
    # Criterion
    MSEloss = torch.nn.MSELoss()
    CEloss = CrossEntropyLoss()
    world_optimiser = torch.optim.Adam(list(encoder.parameters())+list(reconstructor.parameters())+list(dynamics.parameters()))

    for i in range(steps):
        if not state_buffer.empty(): # recieved real world experience
            observations, reward, end = state_buffer.get()
            reward_and_continuation = torch.concat([observations, end, reward], dim=-1)
            l: torch.Tensor = encoder(observations, reward_and_continuation, a0)
            # make action ASAP
            control.drop_traces()
            a, v0 = control.get_action(l, epsilon) # fill buffer with new action
            action_buffer.put(torch.where(end, 1, a)) # Force select Jump to restart ended games

            # train model with real world experience
            observation_recon, RnC_recon = reconstructor(l)
            observation_loss = MSEloss(observations, observation_recon)
            RnC_loss = MSEloss(reward_and_continuation, RnC_recon)
            total_loss = observation_loss + RnC_loss

            if not (RnC0 is None or l0 is None or lpred is None or RnCpred is None): # world model loss
                encode_loss = CEloss(l, lpred.detatch())
                dynam_pred_loss = CEloss(l.detach(), lpred)
                RnC_pred_loss = MSEloss(reward_and_continuation.detach(), RnCpred)
                total_loss += encode_loss + dynam_pred_loss + RnC_pred_loss
            
            total_loss.back() # back propagate loss
            world_optimiser.step()

            # make new predictions
            RnC0 = reward_and_continuation
            l0 = l
            a0 = a
            lpred, RnCpred = dynamics(l, a)
            lr0 = None
                
            
        elif not (RnC0 is None or l0 is None or lpred is None or RnCpred is None or v0 is None): # halucinate
            if (lr0 is None or lr1 is None or vr is None or ar is None): # migrate initialisation variables
                lr0 = l0
                lr1 = lpred
                ar = a0
                vr = v0
                RnCr0 = RnC0
            a, v = control.get_action(lpred, epsilon)
            # td = gamma * torch.max(v).detach() - vr[ar] # Q learning
            td = gamma * v[a] - vr[ar] # SARSA
            control.backprop(td)

            



            
            


                
            
            



