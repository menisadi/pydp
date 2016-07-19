from __future__ import division
from numpy import zeros, sum, sqrt, log
import numpy as np
from numpy.random import laplace
from bounds import log_star
from basicdp import exponential_mechanism
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances as distances
from flat_concave import evaluate


# TODO no need for this method (sklearn has a better one)
def __distances__(points):
    n = len(points)
    neighbourhood = zeros((n, n))
    for i in xrange(n):
        for j in xrange(i):
            neighbourhood[i, j] = neighbourhood[j, i] = euclidean(points[i], points[j])
    return neighbourhood


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


def __promise__(data, domain, eps, delta, failure):
    # TODO docstring
    """

    :param data:
    :param domain:
    :param eps:
    :param delta:
    :param failure:
    :return:
    """
    const = log_star(2 * (domain[1] - domain[0] + 1) * sqrt(data.shape[1]))
    return 8 ** const * 144 * const / eps * log(24 * const / failure / delta)


# the parameter promise should later be removed from the input and be calculated within the function
def find(data, domain, goal_number, failure, eps, delta):
    # TODO docstring
    """

    :param data:
    :param domain:
    :param goal_number:
    :param failure:
    :param eps:
    :param delta:
    :return:
    """
    # domain = min(np.min(data, axis=0)), max(np.max(data, axis=0))
    promise = __promise__(data, domain, eps, delta, failure)
    all_distances = distances(data)
    if __max_average_ball__(0, all_distances, goal_number) + laplace(0, 4/eps, 1) >\
                            goal_number - 2*promise - 4/eps*log(2/failure):
        return 0

    def quality(d, r):
        return min(goal_number - __max_average_ball__(r/2, all_distances, goal_number),
                   __max_average_ball__(r, all_distances, goal_number) - goal_number + 4*promise) / 2

    # TODO must complete those two
    def radius_interval_bounding():
        pass

    def max_radius_in_interval():
        pass

    return evaluate(data, range(domain[1]-domain[0]), quality, promise,
                    0.5, eps, delta,
                    radius_interval_bounding, max_radius_in_interval)

