from time import time, sleep
from OfflineOnline.config.paths import locateOnScreenTargets, crashRecords

def wait_on_img(pyautogui, img_name: str, time_out:float = 10, retry_interal: float = 0.1) -> None:
    full_path = locateOnScreenTargets / img_name
    start_time = time()
    while time()-start_time < time_out:
        try:
            return pyautogui.locateOnScreen(str(full_path), confidence = 0.9, grayscale = True)
        except pyautogui.ImageNotFoundException:
            pass
        sleep(retry_interal)
    pyautogui.screenshot().save(crashRecords / f"wait_on_{img_name}_crash.png")
    raise Exception(f"{full_path} not found on screen in {time_out} seconds")