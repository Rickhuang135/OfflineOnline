from OfflineOnline.config.paths import locateOnScreenTargets, crashRecords
from time import time

time_out = 10
def clean(pyautogui):
    begin_time = time()
    try:
        while time()-begin_time < time_out:
            left, top, width, height =pyautogui.locateOnScreen(str(locateOnScreenTargets / "cross.png"), confidence = 0.9, grayscale = True)
            x = left + width/2
            y = top + height/2
            pyautogui.click(x, y)
            print("closed a thing")
        pyautogui.screenshot().save(crashRecords / f"align_screen_crash.png")
        raise Exception("Unable to close all pollutants")
    except pyautogui.ImageNotFoundException:
        return