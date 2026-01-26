import numpy as np
import asyncio
import time

from .utils.slice_shape import decapitate as decapitate_shape
from .utils.numpy_similarity import compare as compare_nparray
from .utils.check_end import check_end
from .utils.save_crash import save_to_png
from .childprocess import ChildProcess
from .keywords import Words, Actions


class VguiBatch:
    time_out = 10
    similarity_threshold = 0.99

    def __init__(
            self, 
            n_parallel: int, 
            verbose: int = 2, # 0 for silent, 1 for intialise only, 2 for everything
                 ):
        self.time_init = time.time()
        self.time_ready = -1
        self.loop = asyncio.get_running_loop()
        self.verbose = verbose
        self.n_parallel = n_parallel
        self.children: list[ChildProcess] = [None for _ in range(n_parallel)] # type: ignore

    @classmethod
    async def create(
        cls, 
        n_parallel: int, 
        verbose: int = 2, # 0 for silent, 1 for intialise only, 2 for everything
        perform_checks: bool = True, # optional checks that ensure data security
        ):
        instance = cls(n_parallel, verbose)

        # Wait for VGUIs to be ready

        instance.print1(f"preparing {n_parallel} processes")
        instance.print1(f"Ready Processes: ", end="")

        async with asyncio.timeout(cls.time_out):
            async with asyncio.TaskGroup() as tg:
                create_children_tasks = [tg.create_task(ChildProcess.create()) for _ in range(n_parallel)]

        for task in create_children_tasks:
            child = task.result()
            instance.children[child.id] = child
            instance.print1(f"{child.id}, ", end="") 

        instance.print1()

        # wrapping checks with try except blocks to terminate waiting vguis
        try:
            # optional checks
            if perform_checks:
                instance.print1("Performing optional environment checks")
                # horizontal check: ensure all child processes have the same starting image
                arrays = instance.shm_all()
                if n_parallel > 1:
                    example_id = 0 
                    example_array = arrays[0]
                    similarities = compare_nparray(arrays[1:], example_array)
                    for i, similarity in enumerate(similarities):
                        id = i+1
                        if similarity < cls.similarity_threshold:
                            save_to_png(example_array, f"horizontal_match_failed_vgui{example_id}")
                            save_to_png(arrays[id], f"horizontal_match_failed_vgui{id}")
                            raise Exception(f"Numpy inputs from vgui {example_id} and vgui {id} has similarity {similarity}, threshold is {cls.similarity_threshold}")
                    
                # vertical check: play one round on each vgui to check movement and game end dectection
                repeat_inds = np.zeros(1)
                start = time.time()
                # instance.children[1].conn.send(Words.setVerbose)
                # instance.children[1].conn.send(Words.SAVEGAME)
                previous_arr = arrays
                arrays, time_stamps = await instance.fetch()
                time_stamps_prev = np.zeros_like(time_stamps) + start

                while len(repeat_inds) < n_parallel and time.time()-start < instance.time_out:
                    # check if new states match with previous states
                    match_prev = np.all(previous_arr==arrays, axis=decapitate_shape(arrays.shape)) # shape (n_parallel)
                    if match_prev.any():
                        repeat_inds = np.where(match_prev==True)[0]
                        end_condition = check_end(previous_arr[repeat_inds])
                        # all repeating states should be end states
                        if not end_condition.all(): # there exists repeating states which are NOT end states
                            problematic_inds = repeat_inds[np.where(end_condition==False)[0]]
                            times_elapsed_prev = time_stamps_prev - instance.time_init
                            times_elapsed_now = time_stamps - instance.time_init
                            for ind in problematic_inds:
                                save_to_png(previous_arr[ind], f"vertical_match_failed_vgui{ind}_{times_elapsed_prev[ind]:.2f}")
                                save_to_png(arrays[ind], f"vertical_match_failed_vgui{ind}_{times_elapsed_now[ind]:.2f}")
                            raise Exception(f"Consecutive inputs from vgui {repeat_inds} repeated without ending at {np.min(times_elapsed_prev)}-{np.max(times_elapsed_now)}")
                    previous_arr, time_stamps_prev = arrays, time_stamps
                    arrays, time_stamps = await instance.fetch([Actions.Duck for _ in range(n_parallel)])

        except Exception as e:
            instance.end()
            raise e
        instance.time_ready = time.time()
        instance.print1(f"Environment ready in {instance.time_ready - instance.time_init}")

        return instance

    def print1(self, msg = "", **kwargs):
        if self.verbose >= 1:
            print(msg, **kwargs)
        
    def print2(self, msg = "", **kwargs):
        if self.verbose >= 2:
            print(msg, **kwargs)

    def batchsend(self, msg: str):
        for child in self.children:
            child.conn.send(msg)

    def end(self):
        self.batchsend(Words.CLOSEDISPLAYS)
        for child in self.children:
            child.shm.close()

    def shm_all(self):
        return np.stack([c.shm_array for c in self.children])

    async def fetch(self, actions: list[str] | None = None):
        if actions is None:
            actions = [Actions.Jump for _ in range(self.n_parallel)]

        async with asyncio.timeout(self.time_out):
            async with asyncio.TaskGroup() as tg:
                get_time_stamp_task = [tg.create_task(c.send_and_read(a)) for c,a in zip(self.children, actions)]

        time_stamps = np.array([task.result() for task in get_time_stamp_task])
        return self.shm_all(), time_stamps
    