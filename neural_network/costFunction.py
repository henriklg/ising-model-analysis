import numpy as np
from math import log, exp

def f(x, w):
    return 1/(1 + exp(-x.dot(w)))

def costFunction(X, labels, w):
    """
    :param X: (nx1600)
    :param labels: y (nx1)
    :param w: omega, weights (1600x1)
    :return:
    """
    n, p = np.shape(X)
    C = 0
    for i in range(n):
        C += -labels[i]*log(f(X[i, :].dot(w))) - (1-labels[i])*log(1 - f(X[i, :].dot(w)))

    return C