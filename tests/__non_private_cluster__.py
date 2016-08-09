from __future__ import division
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import euclidean_distances as distances
from numpy import where
import matplotlib.pyplot as plt


def find_cluster(data_set, k):
    """

    :param data_set:
    :param k: number of desired points in cluster
    :return:
    """
    distance = distances(data_set)
    for point in data_set:
        point_index = where(data_set == point)[0][0]
        near_k = np.sort(distance[point_index])[k]
        try:
            if near_k < r:
                r, c = near_k, point
        except NameError:
            r, c = near_k, point
    return r, c


# for plotting
def circle(r, phi, p):
    return r*np.cos(phi)+p[0], r*np.sin(phi)+p[1]


def test():
    n = 50
    data = np.random.normal(0, 10, (n, 2))
    r, c = find_cluster(data, 10)
    print r, c
    plt.scatter(*zip(*data))
    phis = np.arange(0, 6.283, 0.01)
    plt.plot(*circle(r, phis, c), c='g', ls='-')
    plt.show()

