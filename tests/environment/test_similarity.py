import numpy as np
from PIL import Image
from os import listdir

from OfflineOnline.config.paths import crashRecords
from OfflineOnline.environment.utils.numpy_similarity import compare

IDENTIFIER = "match_failed_vgui" 

crash_file_names = list(filter(lambda x: IDENTIFIER in x, listdir(crashRecords)))
crash_file_arrays = []
crash_file_ind = []

for file_name in crash_file_names:
    img = Image.open(crashRecords / file_name, mode="r")
    crash_file_arrays.append(np.array(img))
    crash_file_ind.append("".join(file_name.split(IDENTIFIER)))

crash_file_batch = np.stack(crash_file_arrays)

for name1, arr1 in zip(crash_file_ind, crash_file_arrays):
    similarities = compare(crash_file_batch, arr1)
    for name2, sim in zip(crash_file_ind, similarities):
        print(f"similarity between {name2}, {name1} is {sim}")


    