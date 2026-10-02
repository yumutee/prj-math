import numpy as np

c = 0.4        # non-linear damping parameter
alpha = 2.0    # u(0)
beta = -2.0    # u'(0)
T = 15.0       # final time
I = 100        # number of steps
N = 2          # dimension of the first-order system

Y0 = np.array([alpha, beta], dtype=float)   # always float, never int


def F(t, Y):
    Y1, Y2 = Y[0], Y[1]
    return np.array([Y2, c * (1.0 - Y1**2) * Y2 - Y1])
