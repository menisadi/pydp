from __future__ import division
import numpy as np
from numpy.linalg import norm


def __nearest__(x, k, c):
    norms = np.array([[p, norm(c-p)] for p in x])
    sorted_norms = [norms[i][0] for i in np.argsort(norms[:, 1])]
    return sorted_norms[:k]


def find_cluster(data_set, k):
    for point in data_set:
        near_point = __nearest__(data_set, k, point)
        curr_radius = max(norm(p-point) for p in near_point)
        try:
            if curr_radius < r:
                r, c = curr_radius, point
        except NameError:
            r, c = curr_radius, point
    return r, c


# for plotting
def circle(r, phi, p):
    return r*np.cos(phi)+p[0], r*np.sin(phi)+p[1]
