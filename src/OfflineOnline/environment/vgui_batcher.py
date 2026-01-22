from multiprocessing import Process, Pipe, resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time

from .utils.slice_shape import decapitate as decapitate_shape
from .utils.numpy_similarity import compare as compare_nparray
from .utils.check_end import check_end
from .utils.save_crash import save_to_png
from .vgui import Vgui
from .keywords import Words, Actions


class VguiBatch:
    time_out = 10
    similarity_threshold = 0.99

    def __init__(
            self, 
            n_parallel: int, 
            verbose: int = 2, # 0 for silent, 1 for intialise only, 2 for everything
            perform_checks: bool = True, # optional checks that ensure data security
                 ):
        self.time_init = time.time()
        self.verbose = verbose
        self.n_parallel = n_parallel
        self.running = False
        self.pipes: list[Connection] = [None for _ in range(self.n_parallel)]  # type:ignore
        self.shm_arrays: list[np.ndarray] = [None for _ in range(self.n_parallel)]  # type:ignore
        self.shm = []
        # Assign pipes to child processes
        pipes = []
        for i in range(n_parallel):
            parent_conn, child_conn = Pipe()
            pipes.append(parent_conn)
            p = Process(target=Vgui, args=(child_conn,))
            p.start()

        # Wait for VGUIs to be ready
        n_ready = 0
        begin_time = time.time()
        self.print1(f"preparing {n_parallel} processes")
        self.print1(f"Ready Processes: ", end="")
        while n_ready < n_parallel and time.time() - begin_time < self.time_out:
            for pipe in pipes:
                if pipe.poll():
                    msg = pipe.recv()
                    id = int(msg[Words.id])
                    shm = SharedMemory(name = msg[Words.shm_name])
                    resource_tracker.unregister(shm._name, "shared_memory") # type: ignore
                    self.shm.append(shm)
                    shm_array = np.ndarray(
                        msg[Words.shape],
                        dtype = msg[Words.dtype],
                        buffer = shm.buf
                    ) # shape (height, width, color_channels)
                    n_ready += 1
                    self.print1(f"{id}, ", end="") 
                    self.pipes[id] = pipe
                    self.shm_arrays[id] = shm_array
        self.print1()

        # wrapping checks with try except blocks to terminate waiting vguis
        try:
            # check for timed out children
            if None in self.pipes:
                raise Exception("Some processes failed to initialise")
            
            # optional checks
            if perform_checks:
                # horizontal check: ensure all child processes have the same starting image
                arrays, time_stamps = self.fetch()
                if n_parallel > 1:
                    example_id = 0 
                    example_array = arrays[0]
                    similarities = compare_nparray(arrays[1:], example_array)
                    for i, similarity in enumerate(similarities):
                        id = i+1
                        if similarity < self.similarity_threshold:
                            save_to_png(example_array, f"horizontal_match_failed_vgui{example_id}")
                            save_to_png(arrays[id], f"horizontal_match_failed_vgui{id}")
                            raise Exception(f"Numpy inputs from vgui {example_id} and vgui {id} has similarity {similarity}, threshold is {self.similarity_threshold}")
                    
                # vertical check: play one round on each vgui to check movement and game end dectection
                repeat_inds = np.zeros(1)
                start = time.time()
                # self.pipes[0].send(Words.setVerbose)
                # self.pipes[0].send(Words.SAVEGAME)
                previous_arr, time_stamps_prev = arrays, time_stamps
                arrays, time_stamps = self.fetch()
                while len(repeat_inds) < n_parallel and time.time()-start < self.time_out:
                    # check if new states match with previous states
                    match_prev = np.all(previous_arr==arrays, axis=decapitate_shape(arrays.shape)) # shape (n_parallel)
                    if match_prev.any():
                        repeat_inds = np.where(match_prev==True)[0]
                        end_condition = check_end(previous_arr[repeat_inds])
                        # all repeating states should be end states
                        if not end_condition.all(): # there exists repeating states which are NOT end states
                            problematic_inds = repeat_inds[np.where(end_condition==False)[0]]
                            times_elapsed_prev = time_stamps_prev - self.time_init
                            times_elapsed_now = time_stamps - self.time_init
                            for ind in problematic_inds:
                                save_to_png(previous_arr[ind], f"vertical_match_failed_vgui{ind}_{times_elapsed_prev[ind]:.2f}")
                                save_to_png(arrays[ind], f"vertical_match_failed_vgui{ind}_{times_elapsed_now[ind]:.2f}")
                            raise Exception(f"Consecutive inputs from vgui {repeat_inds} repeated without ending at {np.min(times_elapsed_prev)}-{np.max(times_elapsed_now)}")
                    previous_arr, time_stamps_prev = arrays, time_stamps
                    arrays, time_stamps = self.fetch([Actions.Duck for _ in range(n_parallel)])

        except Exception as e:
            self.end()
            raise e
        self.time_ready = time.time()
        self.print1(f"Environment ready in {self.time_ready - self.time_init}")

    def print1(self, msg = "", **kwargs):
        if self.verbose >= 1:
            print(msg, **kwargs)
        
    def print2(self, msg = "", **kwargs):
        if self.verbose >= 2:
            print(msg, **kwargs)

    def batchsend(self, msg: str):
        for conn in self.pipes:
            conn.send(msg)

    def end(self):
        self.batchsend(Words.CLOSEDISPLAYS)
        for shm in self.shm:
            shm.close()

    def fetch(self, actions: list[str] | None = None):
        if actions is None:
            actions = [Actions.Jump for _ in range(self.n_parallel)]
        for conn, action in zip(self.pipes, actions):
            conn.send(action)

        time_stamps_arr = [None for _ in range(self.n_parallel)]
        start = time.time()
        while time.time()-start < self.time_out and None in time_stamps_arr:
            for i, time_stamp in enumerate(time_stamps_arr):
                if time_stamp is None:
                    c_pipe = self.pipes[i]
                    if c_pipe.poll():
                        time_stamps_arr[i] = c_pipe.recv()
        if None in time_stamps_arr:
            missing_index = time_stamps_arr.index(None)
            raise Exception(f"Process {missing_index} dropped after {time.time()-start:2f}s during image retrieval")
        
        time_stamps = np.array(time_stamps_arr)
        frames = np.stack(self.shm_arrays)
        return frames, time_stamps
    