import numpy as np

from .discretizeAB import discretize
from .generate_polynomials import p

class Memory:
    def __init__(self, window_size: int, dimensions: int = 20):
        self.values = np.zeros(dimensions)
        self.window_size = window_size
        self.dimensions = dimensions

        # build matrices
        raw_B = (-1 ) ** np.arange(dimensions) * (2*np.arange(dimensions)+1) / window_size

        rows = []
        for i in range(dimensions):
            res = np.ones(dimensions)*-1
            res[:i] = (-1 ) ** (np.arange(i)+i+1)
            rows.append(res*(2*i+1))
        raw_A = np.stack(rows) / window_size
        print(raw_A)

        # discretize matrices
        self.A, self.B = discretize(raw_A, raw_B)
        print(self.A)
        print(self.B)
        print("Initiallisation finished")

    def update(self, new_input):
        print(self.A@self.values)
        print(self.B*new_input)
        self.values = self.A@self.values + self.B*new_input
        print(self.values)

    def __getitem__(self, index):
        return self.values[index]

    def reconstruct(self):
        def approx(x: np.ndarray): # expects values betwee -1 and 1
            stack = p(x, self.dimensions)
            return (stack.T@self.values)
        return approx