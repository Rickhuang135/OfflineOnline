from .vgui_batcher import VguiBatch
from .utils.check_end import check_end

class Envrionment(VguiBatch):
    def __init__(
            self, 
            n_parallel: int, 
            verbose: int = 2, # 0 for silent, 1 for intialise only, 2 for everything
                 ):
        super().__init__(n_parallel, verbose)
    
    async def get(self, actions: list[str] | None = None):
        observations, time_stamps = await self.fetch(actions)
        end = check_end(observations)
        reward = end * -1
        return observations, reward, end, time_stamps