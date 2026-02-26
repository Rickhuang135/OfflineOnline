import numpy as np

from .discretizeAB import discretize
from .generate_polynomials import phi as p

class Memory:
    def __init__(self, window_size: int, number_states: int = 1, degree_approx: int = 20):
        self.values = np.zeros((degree_approx, number_states))
        self.window_size = window_size
        self.degree_approx = degree_approx

        # build matrices
        i = np.arange(degree_approx)
        raw_B = (-1 ) ** i * np.sqrt(2*(2*i+1)) / window_size

        i, j = np.indices((degree_approx, degree_approx))
        power_condition = ((i+j)%2 == 0) + (j>i)
        raw_A = np.where(power_condition, -1, 1)
        scale_terms = np.sqrt((2*i+1)*(2*j+1))
        raw_A = raw_A * scale_terms / window_size

        # discretize matrices
        self.A, self.B = discretize(raw_A, raw_B)
        self.B = self.B.reshape(-1,1)

    def update(self, array):
        self.values = self.A@self.values + self.B@array.reshape(1,-1)

    def reconstruct(self, x: np.ndarray):
        stack = p(x, self.degree_approx)
        return (stack.T@self.values).T