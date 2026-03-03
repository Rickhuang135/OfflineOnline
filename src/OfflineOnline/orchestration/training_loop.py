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
    o0 = None
    r0 = None
    a0 = torch.zeros(batch_size, device=device)
    lmeans_0 = None
    lstd_0 = None
    lr0 = None
    lr1 = None
    MSEloss = torch.nn.MSELoss()
    CEloss = CrossEntropyLoss()

    def get_reconstruction_loss(latent_state: torch.Tensor): # don't use on prior dynamics predictions
        observation, RnC = reconstructor(latent_state)
        

    for i in range(steps):
        if not state_buffer.empty(): # recieved real world experience
            observations, reward, end = state_buffer.get()
            reward_and_continuation = torch.concat([observations, end, reward], dim=-1)
            lmeans, lstds = encoder(observations, reward_and_continuation, a0)
            # make action ASAP
            a = control.get_action(torch.concat([lmeans, lstds], dim=-1), epsilon, write_to_window=False) # don't train with real world experience
            action_buffer.put(torch.where(end, 1, a)) # Force select Jump to restart ended games

            # train model with real world experience
            observation_recon, RnC_recon = reconstructor(l)
            observation_loss = MSEloss(observations, observation_recon)
            RnC_loss = MSEloss(reward_and_continuation, RnC_recon)
            reconstruct_loss = observation_loss + RnC_loss

            if o0 is None or r0 is None: # first observation, no training
                o0 = observations
                r0 = reward
                l0 = l
            else: # train world model
                prediction_loss = CEloss()
                
            
        elif l0 is not None: # halucinate
            if lr0 is None:
                lr0 = l0
