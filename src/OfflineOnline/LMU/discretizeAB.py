import numpy as np

from .e_approximates import pade_approx

def discretize(A: np.ndarray, B: np.ndarray):
    approximation_degree = A.shape[0]
    block_mat = np.zeros((approximation_degree+1, approximation_degree+1))
    block_mat[:-1, :-1] = A
    block_mat[:-1, -1] = B
    block_res = pade_approx(block_mat, 100, 100) # assumes Δt = 1
    Ad = block_res[:-1, :-1]
    Bd = block_res[:-1, -1]
    return Ad, Bd

def euler_discretize(A: np.ndarray, B: np.ndarray, delta_t: float = 1):
    Ad = np.identity(A.shape[0]) + A*delta_t
    Bd = B*delta_t
    return Ad, Bd