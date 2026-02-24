from matplotlib import pyplot as plt
import numpy as np

from OfflineOnline.LMU.e_approximates import taylor_approx, pade_approx


def visual_test(degree_approx = 5, test_range = 10):
    x = np.arange(0, test_range, 0.1)
    y = np.e**x

    y_approx_t = taylor_approx(x, degree_approx)
    y_approx_p = pade_approx(x, degree_approx//2+1, degree_approx//2-1)

    plt.plot(x, y, label="e^x")
    plt.plot(x, y_approx_t, label="taylor")
    plt.plot(x, y_approx_p, label="pade")
    plt.legend()
    plt.show()

def matrix_test(degree_approx = 5, test_range = 10):
    A = np.array([[0,-1],[1,0]], dtype=np.float64)
    error_taylor = []
    error_pade = []
    x = np.arange(test_range)
    for t in x:
        At = np.array([
            [np.cos(t), -np.sin(t)],
            [np.sin(t), np.cos(t)]
        ])
        At_taylor = taylor_approx(A*t, degree_approx)
        At_pade = pade_approx(A*t, degree_approx//2+1, degree_approx//2-1)
        error_taylor.append(((At-At_taylor)**2).mean())
        error_pade.append(((At-At_pade)**2).mean())

    # plt.plot(x, error_taylor, label="taylor")
    plt.plot(x, error_pade, label="pade")
    plt.legend()
    plt.show()

matrix_test(10,30)
# visual_test()