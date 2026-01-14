
from .vgui import Vgui
from .keywords import Words

from multiprocessing import Process, Pipe
import time

class VguiBatch:
    time_out = 10

    def __init__(
            self, 
            n_parallel: int, 
            verbose: int = 2, # 0 for silent, 1 for intialise only, 2 for everything
                 ):
        self.verbose = verbose
        self.n_parallel = n_parallel
        self.pipes = []

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
                    n_ready += 1
                    self.print1(f"{msg}, ", end="")
                    self.pipes.append((msg, pipe))
        if n_ready < n_parallel:
            raise Exception("Some processes failed to initialise")

    def print1(self, msg, **kwargs):
        if self.verbose >= 1:
            print(msg, **kwargs)
        
    def print2(self, msg, **kwargs):
        if self.verbose >= 2:
            print(msg, **kwargs)

    def batchsend(self, msg: str):
        for id, conn in self.pipes:
            conn.send(msg)
        
    def start(self):
        self.batchsend(Words.STARTGAMES)

    def end(self):
        self.batchsend(Words.CLOSEDISPLAYS)