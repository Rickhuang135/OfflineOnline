import torch
import torch.multiprocessing as mp

from OfflineOnline.config.constants import game_canvas, gray_scale
from OfflineOnline.environment.keywords import Actions
from OfflineOnline.environment import Envrionment

def interpret_str(actions: torch.Tensor):
    res = []
    for action in actions:
        match action:
            case 0: res.append(Actions.Nothing)
            case 1: res.append(Actions.Jump)
            case 2: res.append(Actions.Duck)
            case _: raise Exception(f"Unexpected action {action}")
    return res


async def environment_loop(env: Envrionment, action_buffer: mp.Queue, state_buffer: mp.Queue, steps: int, device = None):
    cached_actions = None
    for _ in range(steps):
        if not action_buffer.empty():
            cached_actions = interpret_str(action_buffer.get())
        observations, reward, end, time_stamps = await env.get(cached_actions)
        # observations have shape (batch, screen_width, screen_height, channels)
        expected_flattened_size = game_canvas[2] * game_canvas[3]
        expected_flattened_size *= 3 if gray_scale else 1
        tc_observations = torch.Tensor(observations.reshape(-1, expected_flattened_size), device=device)
        tc_end = torch.Tensor(end, device=device)
        tc_reward = torch.Tensor(reward, device=device)
        if state_buffer.full():
            while not state_buffer.empty():
                _ = state_buffer.get_nowait()
        state_buffer.put((tc_observations, tc_reward, tc_end))


        