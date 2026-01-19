import numpy as np

def compare(img1: np.ndarray, img2: np.ndarray) -> float:
    d = np.abs(img1 - img2).ravel()
    largest_possible_difference = 255 * len(d)
    difference_percentage = np.sum(d) / largest_possible_difference
    return 1 - difference_percentage