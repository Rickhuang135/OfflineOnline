from pyvirtualdisplay import Display
import os
import subprocess
import time
from .localpaths import Screenshots
from .utils.virtual_user import assign_directory
from .utils.suppress_warning import filter_proc
from .utils.wait_on_img import wait_on_img
from .keywords import Words

class Vgui:
    size = 670, 500

    def __init__(self, conn):
        self.img_path = Screenshots
        self.saving_index = 0
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
            ["chromium", "--disable-gpu", "--new-window", f"--user-data-dir={assign_directory(self.id)}"],
            env = self.env,
            stderr = subprocess.PIPE,
            text = True
        )
        filter_proc(proc)
        wait_on_img(pyautogui, "Google.png", 4)
        pyautogui.typewrite("chrome://dino")
        pyautogui.typewrite(["enter"])
        wait_on_img(pyautogui, "dino_init.png", 2)


        self.conn.send(self.id)
        return self.get_instruction()

    def get_instruction(self):
        msg = self.conn.recv()
        match msg:
            case Words.STARTGAMES:
                self.game_loop()
            case Words.CLOSEDISPLAYS:
                self.end()
            case _:
                raise Exception(f"vgui {self.id} received unknown instruction {msg}")

    def save_image(self, img):
        img.save(self.img_path / f"vgui{self.id}_{self.saving_index}.png")
        self.saving_index += 1

    def game_loop(self):
        # Start game
        self.controller.typewrite(" ")
        for _ in range(7):
            frame = self.controller.screenshot()
            self.save_image(frame)
            time.sleep(1)

        return self.get_instruction()

    def end(self):
        self.controller.hotkey("alt", "f4")
        time.sleep(1)
        self.display.stop()