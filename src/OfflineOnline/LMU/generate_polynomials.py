import numpy as np

def p(x, dimensions):
    series = [np.ones_like(x), x]
    for n in range(2, dimensions):
        t1 = (2*n-1)*x*series[n-1]
        t2 = -(n-1)*series[n-2]
        series.append((t1+t2)/n)
    return np.stack(series)

def phi(x, dimensions):
    series = p(x, dimensions)
    coefficients = np.sqrt((np.arange(0, dimensions) * 2 + 1)/2)
    temp_series = series.T
    temp_series[:, np.arange(dimensions)] *= coefficients
    return temp_series.T