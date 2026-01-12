from pyvirtualdisplay import Display
import os
import subprocess
import time
from .localpaths import Images
from .utils.virtual_user import assign_directory

class Vgui:
    size = 670, 500

    def __init__(self, conn):
        self.img_path = Images
        self.saving_index = 0
        self.conn = conn

        self.display = Display(visible=False, size=self.size)
        self.display.start()

        # Set correct environment 
        self.env = os.environ.copy()
        self.id = int(self.env["DISPLAY"].strip(":"))
        self.env.pop("WAYLAND_DISPLAY", None)
        self.env["XDG_SESSION_TYPE"] = "x11"

        import pyautogui # Import AFTER virtual display is active
        self.controller = pyautogui
        # Open google chrome
        subprocess.Popen(
            ["chromium", "--disable-gpu", "--new-window", f"--user-data-dir={assign_directory(self.id)}"],
            env=self.env,
        )
        time.sleep(4) # TODO: add more robust loading detection system
        pyautogui.typewrite("chrome://dino")
        pyautogui.typewrite(["enter"])
        time.sleep(1)
        print(f"process {self.id} successfully prepared")

        self.get_instruction()

    def get_instruction(self):
        msg = self.conn.recv()
        match msg:
            case "start":
                self.game_loop()
            case _:
                raise Exception(f"vgui {self.id} received unknown instruction {msg}")

    def save_image(self, img):
        img.save(self.img_path / f"vgui{self.id}_{self.saving_index}.png")
        self.saving_index += 1

    def game_loop(self):
        # Start game
        self.controller.typewrite(" ")
        for _ in range(5):
            frame = self.controller.screenshot()
            self.save_image(frame)
            time.sleep(1)
        self.end()


    def end(self):
        self.controller.hotkey("alt", "f4")
        time.sleep(1)
        self.display.stop()