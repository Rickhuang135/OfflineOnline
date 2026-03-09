import asyncio

from OfflineOnline.config.constants import game_canvas
from OfflineOnline.config.hyperparams import latent_state_size, n_parallel
from OfflineOnline.config.device import DEVICE

from .environment_loop import environment_loop, Envrionment
from OfflineOnline.encoder import Encoder
from OfflineOnline.encoder.reconstruct import Reconstruct
from OfflineOnline.worldmodel import DynamicsModel
from OfflineOnline.control import SarsaLambda

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
    

    env = await task_create_env

asyncio.run(main())