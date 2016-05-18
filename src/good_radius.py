from numpy import zeros
from numpy.linalg import norm
import numpy as np


def __neighbours__(points, radius):
    n = len(points)
    close_enough = zeros((n, n))
    for i in xrange(n):
        for j in xrange(i):
            if norm(points[i]-points[j]) <= radius:
                close_enough[i, j] = close_enough[j, i] = 1
    return close_enough


def find(data, t, failure, eps, delta):
    return


