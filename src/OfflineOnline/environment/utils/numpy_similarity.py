import numpy as np

from .slice_shape import decapitate

def compare(batch: np.ndarray, target: np.ndarray):
    n_parallel = batch.shape[0]
    differences = np.abs(batch.astype(np.int8) - target.astype(np.int8))
    largest_possible_differences = 255 * differences.size / n_parallel
    difference_percentages = np.sum(differences, axis = decapitate(batch.shape)) / largest_possible_differences
    return 1- difference_percentages
