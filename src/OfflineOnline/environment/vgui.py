from multiprocessing.shared_memory import SharedMemory
from multiprocessing.connection import Connection
from pyvirtualdisplay import Display
from PIL.Image import Image
import numpy as np
import os
import subprocess
import time

from OfflineOnline.config.paths import Screenshots
from OfflineOnline.config.constants import window_size, game_canvas
from .utils.virtual_user import assign_directory
from .utils.suppress_warning import filter_proc
from .utils.wait_on_img import wait_on_img
from .keywords import Words, Actions
# from .utils.align_screen import clean

class Vgui:
    size = window_size # (width, height) of virtual display screen
    region = game_canvas # (left, top, width, height) of captured game canvas

    def __init__(self, conn: Connection):
        self.saving_index = 0
        self.saving = False # whether all recorded states are being saved
        self.img_path = Screenshots
        self.conn = conn
        self.display = Display(visible=False, size=self.size, use_xauth=True)
        self.display.start()

        # Set correct environment 
        self.env = os.environ.copy()
        self.id = int(self.env["DISPLAY"].strip(":"))
        self.env.pop("WAYLAND_DISPLAY", None)
        self.env["XDG_SESSION_TYPE"] = "x11"

        import pyautogui # Import AFTER virtual display is active
        self.controller = pyautogui
        # Open google chrome
        proc = subprocess.Popen(
            [
                "chromium", 
                "--disable-gpu", 
                "--new-window", 
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-backgrounding-occluded-windows",
                "--disable-renderer-backgrounding",
                "--disable-features=CalculateNativeWinOcclusion",
                f"--window-size={self.size[0]},{self.size[1]}",
                "--window-position=0,0",
                f"--user-data-dir={assign_directory(self.id)}"
                ],
            env = self.env,
            stderr = subprocess.PIPE,
            text = True
        )
        filter_proc(proc)

        # Load dino game
        wait_on_img(pyautogui, "Google.png", 4)
        pyautogui.typewrite("chrome://dino")
        pyautogui.typewrite(["enter"])
        wait_on_img(pyautogui, "dino_init.png", 2)
        # clean(pyautogui)

        # Share buffer and await further instruction
        frame = self.get_frame()
        self.shm = SharedMemory(create=True, size=frame.nbytes)
        self.shm_array = np.ndarray(frame.shape, dtype=frame.dtype, buffer=self.shm.buf)
        self.shm_array[:] = frame

        self.conn.send({
            Words.id: self.id,
            Words.shm_name: self.shm.name,
            Words.shape: frame.shape,
            Words.dtype: str(frame.dtype),
        })
        return self.get_instruction()

    def get_instruction(self):
        msg = self.conn.recv()
        match msg:
            case Actions.Jump:
                self.action_jump()
                self.send_frame()
            case Actions.Duck:
                self.action_duck()
                self.send_frame()
            case Actions.Nothing:
                self.send_frame()
            case Words.SAVEGAME:
                self.saving = True
                print("Now saving images")
            case Words.CLOSEDISPLAYS:
                self.end()
                return 
            case _:
                raise Exception(f"vgui {self.id} received unknown instruction {msg}")
        return self.get_instruction()

    def get_frame(self) -> np.ndarray:
        img = self.controller.screenshot( region = self.region )
        if self.saving:
            self.save_image(img)
        img_arr = np.array(img, dtype=np.uint8, copy=True)
        return img_arr
    
    def action_jump(self):
        self.controller.press('up')

    def action_duck(self):
        self.controller.press('down')

    def send_frame(self):
        self.shm_array[:] = self.get_frame()
        self.conn.send(time.time())

    def save_image(self, img: Image):
        img.save(self.img_path / f"vgui{self.id}_{self.saving_index}.png")
        self.saving_index += 1

    def end(self):
        self.controller.hotkey("alt", "f4")
        time.sleep(1)
        self.display.stop()
        self.shm.unlink()
        self.shm.close()