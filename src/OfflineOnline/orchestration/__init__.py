import asyncio
import torch.multiprocessing as mp
from time import time

from OfflineOnline.config.constants import game_canvas
from OfflineOnline.config.hyperparams import latent_state_size, n_parallel, n_steps
from OfflineOnline.config.device import DEVICE
from OfflineOnline.encoder import Encoder
from OfflineOnline.encoder.reconstruct import Reconstruct
from OfflineOnline.worldmodel import DynamicsModel
from OfflineOnline.control import SarsaLambda

from .environment_loop import environment_loop, Envrionment
from .training_loop import train

raw_input_width = game_canvas[2]
raw_input_height = game_canvas[3]
nx = raw_input_width*raw_input_height


async def main():
    loop = asyncio.get_running_loop()
    task_create_env = asyncio.create_task(Envrionment.create(n_parallel))
    encoder = Encoder(nx, latent_state_size)
    reconstructor = Reconstruct(latent_state_size, nx)
    dynamics = DynamicsModel(latent_state_size)
    control = SarsaLambda(latent_state_size, 3, 5, 0.8, n_parallel)
    action_buffer = mp.Queue()
    state_buffer = mp.Queue()

    env = await task_create_env

    print("starting training")
    start_time = time()
    train_task = asyncio.create_task(train(encoder, reconstructor, dynamics, control, action_buffer, state_buffer, n_steps))
    env_task = asyncio.create_task(environment_loop(env, action_buffer, state_buffer, n_steps, device=DEVICE))

    await train_task, env_task
    print(f"training finished after {time()-start_time} seconds")

if __name__ == "__main__":
    asyncio.run(main())