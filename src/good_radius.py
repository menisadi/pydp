from __future__ import division
from numpy import zeros, sum, log
import numpy as np
from numpy.random import laplace
from basicdp import exponential_mechanism_big
from sklearn.metrics.pairwise import euclidean_distances as distances


def __max_average_ball__(radius, hood, t):
    # TODO docstring
    """

    :param radius:
    :param hood:
    :param t:
    :return:
    """
    close_points = (hood <= radius).astype(int)
    a = sum(close_points, axis=1).argsort()[t:]
    return sum(min(sum(i), t) for i in close_points[a]) / t


# the parameter promise should later be removed from the input and be calculated within the function
def find(data, domain, goal_number, failure, eps):
    # TODO docstring
    """

    :param data:
    :param domain:
    :param goal_number:
    :param failure:
    :param eps:
    :return:
    """
    # domain = min(np.min(data, axis=0)), max(np.max(data, axis=0))
    all_distances = distances(data)
    # TODO since we are not using rec_concave - is this necessary?
    # TODO if it is - should the bound stay like this?
    if __max_average_ball__(0, all_distances, goal_number) + laplace(0, 4/eps, 1) >\
                            goal_number - 4/eps*log(2/failure):
        return 0

    def q(d, r):
        return __max_average_ball__(r, all_distances, goal_number)

    return exponential_mechanism_big(data, range(domain[1]-domain[0]), q, eps)

