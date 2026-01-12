from .vgui import Vgui

from multiprocessing import Process, Pipe
import time

n_parallel = 2

pipes = []
for i in range(n_parallel):
    parent_conn, child_conn = Pipe()
    pipes.append(parent_conn)
    p = Process(target=Vgui, args=(child_conn,))
    p.start()

time.sleep(7)
for conn in pipes:
    print("sending message")
    conn.send("start")



