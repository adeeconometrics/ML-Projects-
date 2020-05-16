import numpy as np


def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])  # initialization
    for i in range(x.shape[0]):  # traverse through i-axis
        for j in range(x.shape[1]):  # traverse through j-axis
            z[i] += x[i, j] * y[j]

    return z
