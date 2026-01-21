import numpy as np
from PIL import Image
from os import listdir

from OfflineOnline.environment.utils.check_end import check_end
from pathlib import Path

path = Path(__file__).resolve().parent / "check_end_imgs"
file_names = listdir(path)
imgs = [np.array(Image.open(path / file, "r")) for file in file_names]

res = check_end(np.stack(imgs), True)
for name, res in zip(file_names, res):
    if res:
        print(f"{name} contains a terminal position")
    else:
        print(f"{name} is ongoing")