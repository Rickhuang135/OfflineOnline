from pathlib import Path
from PIL import Image
import numpy as np

from OfflineOnline.config.paths import locateOnScreenTargets
from OfflineOnline.config.constants import game_over_location # (left, top, width, height)
from .numpy_similarity import compare

THRESHOLD = 0.99
FILE_NAME = "gameover.array"
THIS_DIR = Path(__file__).resolve().parent
location = THIS_DIR / FILE_NAME
file_dtype = np.uint8
left, top, width, height = game_over_location
right = left+width
bottom = top+height

def extract():
    # extract image as numpy array
    target = locateOnScreenTargets / "game_over1.png"
    img = np.array(Image.open(target, "r"), dtype = file_dtype)
    result = img[top:bottom, left:right] # shape (height, width, color_channels)

    return result

try:
    target = np.fromfile(location, dtype = file_dtype) # shape (height, width, color_channels)
    target = target.reshape(height, width, -1)
except FileNotFoundError:
    target = extract()
    target.tofile(location)
    # Image.fromarray(target).save(str(THIS_DIR / "gameover.png"))

def check_end(batch: np.ndarray, print_similarity = False) -> np.ndarray:
    check_regions = batch[:, top:bottom, left:right]
    similarity = compare(check_regions, target)
    if print_similarity:
        print(similarity)
    return similarity > THRESHOLD