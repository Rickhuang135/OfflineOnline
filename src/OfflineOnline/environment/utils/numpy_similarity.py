import numpy as np

def compare(batch: np.ndarray, target: np.ndarray):
    n_parallel = batch.shape[0]
    nd = len(batch.shape)
    sum_axis = tuple(range(1,nd))
    differences = np.abs(batch.astype(np.int8) - target.astype(np.int8))
    largest_possible_differences = 255 * differences.size / n_parallel
    difference_percentages = np.sum(differences, axis = sum_axis) / largest_possible_differences
    return 1- difference_percentages
