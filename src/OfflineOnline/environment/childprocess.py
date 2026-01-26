from multiprocessing import Process, Pipe, resource_tracker
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import asyncio

from .vgui import Vgui
from .keywords import Words

class ChildProcess:
    def __init__(self):
        self.id = -1
        self.ready = False
        self.running = False
        self.conn, child_conn = Pipe()
        self.shm_array: np.ndarray = None # type:ignore
        self.shm: SharedMemory = None # type:ignore
        self.p = Process(target = Vgui, args=(child_conn,))

        self.loop = asyncio.get_running_loop()
        self.loop.add_reader(self.conn.fileno(), self._handle_read)

        self.current_future = self.loop.create_future()
        self.p.start()
        
    @classmethod
    async def create(cls):
        instance = cls()
        msg = await instance.current_future # type:ignore
        instance.current_future = None # clean up
        instance.id = int(msg[Words.id])
        shm = SharedMemory(name = msg[Words.shm_name])
        resource_tracker.unregister(shm._name, "shared_memory") # type: ignore
        instance.shm = shm
        shm_array = np.ndarray(
            msg[Words.shape],
            dtype = msg[Words.dtype],
            buffer = shm.buf
        ) # shape (height, width, color_channels)
        instance.shm_array = shm_array
        return instance

    def _handle_read(self):
        if (
            self.current_future and 
            not self.current_future.done() and 
            self.conn.poll()
            ):
            self.current_future.set_result(self.conn.recv())

    async def send_and_read(self, msg):
        if self.current_future is not None:
            raise RuntimeError(f"Communicator for Child process {self.id} is already awaiting a response")

        self.current_future = self.loop.create_future()
        self.conn.send(msg)
        res = await self.current_future
        self.current_future = None
        return res