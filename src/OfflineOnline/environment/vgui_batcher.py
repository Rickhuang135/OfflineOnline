from multiprocessing import Process, Pipe, resource_tracker
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time

from .utils.numpy_similarity import compare
from .utils.save_crash import save_to_png
from .vgui import Vgui
from .keywords import Words, Actions


class VguiBatch:
    time_out = 10
    similarity_threshold = 0.95

    def __init__(
            self, 
            n_parallel: int, 
            verbose: int = 2, # 0 for silent, 1 for intialise only, 2 for everything
            perform_checks: bool = True, # optional checks that ensure data security
                 ):
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
                    )
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
                example_id = 0 
                example_array = self.shm_arrays[0]
                for id, shm_array in enumerate(self.shm_arrays[1:]):
                    similarity = compare(example_array, shm_array)
                    if similarity < self.similarity_threshold:
                        save_to_png(example_array, f"horizontal_match_failed_vgui{example_id}")
                        save_to_png(shm_array, f"horizontal_match_failed_vgui{id}")
                        raise Exception(f"Numpy inputs from vgui {example_id} and vgui {id} has similarity {similarity}, threshold is {self.similarity_threshold}")
                    
                # vertical check: play one round on each vgui to check movement and game end dectection
                start = time.time()
                self.get()
                while time.time()-start < 7:
                    # self.get([Actions.Duck for _ in range(self.n_parallel)])
                    self.get()
                    time.sleep(0.1)
        except Exception as e:
            self.end()
            raise e

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

    def get(self, actions: list[str] | None = None):
        if actions is None:
            actions = [Actions.Jump for _ in range(self.n_parallel)]
        for conn, action in zip(self.pipes, actions):
            conn.send(action)

        time_stamps = [None for _ in range(self.n_parallel)]
        start = time.time()
        while time.time()-start < self.time_out and None in time_stamps:
            for i, time_stamp in enumerate(time_stamps):
                if time_stamp is None:
                    c_pipe = self.pipes[i]
                    if c_pipe.poll():
                        time_stamps[i] = c_pipe.recv()
        if None in time_stamps:
            missing_index = time_stamps.index(None)
            raise Exception(f"Process {missing_index} dropped after {time.time()-start:2f}s during image retrieval")
        return time_stamps

