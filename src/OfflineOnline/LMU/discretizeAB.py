import numpy as np

from .e_approximates import pade_approx

def discretize(A: np.ndarray, B: np.ndarray, delta_t: float = 1):
    height = A.shape[0] + 1
    width = A.shape[1] + 1
    block_mat = np.zeros((width,height))
    block_mat[:-1, :-1] = A 
    block_mat[:-1, -1] = B
    block_res = pade_approx(block_mat * delta_t, 50, 50)
    Ad = block_res[:-1, :-1]
    Bd = block_res[:-1, -1]
    return Ad, Bd

def euler_discretize(A: np.ndarray, B: np.ndarray, delta_t: float = 1):
    Ad = np.identity(A.shape[0]) + A*delta_t
    Bd = B*delta_t
    return Ad, Bd