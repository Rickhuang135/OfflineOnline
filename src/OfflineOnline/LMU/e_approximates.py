from math import factorial
import numpy as np

def PowerGenerator(x: np.ndarray, power: int):
    res = np.ones_like(x)
    for _ in range(power+1):
        yield res
        res*=x

def MatrixPowerGenerator(x: np.ndarray, power:int):
    res = np.identity(x.shape[0], dtype=np.float128)
    for _ in range(power+1):
        yield res
        res@=x

def taylor_approx(x: np.ndarray, degree: int = 5):
    result = np.zeros_like(x)
    # check if is a square matrix
    if len(x.shape) == 2 and x.shape[0] == x.shape[1]:
        gen = MatrixPowerGenerator(x, degree)
    else:
        gen = PowerGenerator(x, degree)
    for i, xcurrent in enumerate(gen):
        result += xcurrent / factorial(i)
    return result

def pade_approx(x: np.ndarray, m: int=5, n: int=5) -> np.ndarray:
    x = np.astype(x, np.float128)
    px = np.zeros_like(x, dtype=np.float128)
    qx = np.zeros_like(x, dtype=np.float128)
    if len(x.shape) == 2 and x.shape[0] == x.shape[1]:
        genM = MatrixPowerGenerator(x, m)
        genN = MatrixPowerGenerator(x, n)
        res_operation = lambda px, qx: px@np.linalg.inv(qx)
    else:
        genM = PowerGenerator(x, m)
        genN = PowerGenerator(x, n)
        res_operation = lambda px, qx: px/qx

    for i, xcurrent in enumerate(genM):
        numerator = factorial(m+n-i) * factorial(m)
        denominator = factorial(m+n) * factorial(i) * factorial(m-i)
        px += xcurrent * numerator/denominator

    for i, xcurrent in enumerate(genN):
        numerator = factorial(m+n-i) * factorial(n)
        denominator = factorial(m+n) * factorial(i) * factorial(n-i)
        abs_res = xcurrent * numerator/denominator
        qx += abs_res if i%2==0 else abs_res*-1

    return res_operation(np.astype(px, np.float64),np.astype(qx, np.float64))