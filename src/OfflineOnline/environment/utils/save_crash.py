from PIL import Image
import numpy as np

from OfflineOnline.config.paths import crashRecords

def save_to_png(img: Image.Image | np.ndarray, name:str):
    path = str(crashRecords / (name + ".png"))
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path)
    print(f"saved image to {path}")