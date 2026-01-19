from os import path
from os import mkdir
from OfflineOnline.config.paths import VirtualUsers

def assign_directory(window_id : int):
    target = VirtualUsers / f"user{window_id}"
    if not path.isdir(target):
        mkdir(target)
    return target