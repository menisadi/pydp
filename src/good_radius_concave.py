# the original good-radius algorithm uses rec_concave mechanism
# because of utility problem we replaced it with exponential-mechanics
# for completeness of the implantation we include also a version which uses rec_concave
from __future__ import division
from numpy import zeros, sum, sqrt, log, log2, arange, ceil
import numpy as np
from numpy.random import laplace
from bounds import log_star
from basicdp import exponential_mechanism
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import euclidean_distances as distances
from flat_concave import evaluate


def __max_average_ball__(radius, hood, t):
    """
    Quality function
    Used in the procedure 'find' as the basis of the concave-quality-function
    :param radius: possible radius to qualify
    :param hood: distance matrix of all the points in the data. two-dimensional matrix.
    :param t: number of desired points in the cluster (that 'find' is looking for)
    :return: the maximum average number of points in t different balls of radius r,
    when for balls with more than t points we take the value t (min(amount,t))
    """
    close_points = (hood <= radius).astype(int)
    a = sum(close_points, axis=1).argsort()[-t:]
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
    const = log_star(2 * (domain + 1) * sqrt(data.shape[1]))
    return 8 ** const * 144 * const / eps * log(24 * const / failure / delta)


# the parameter promise should later be removed from the input and be calculated within the function
def find(data, goal_number, failure, eps, delta, promise=-1):
    # TODO docstring
    """

    :param data:
    :param goal_number:
    :param failure:
    :param eps:
    :param delta:
    :param promise:
    :return:
    """
    domain = abs(max(np.max(data, axis=0)) - min(np.min(data, axis=0)))
    if promise == -1:
        promise = __promise__(data, domain, eps, delta, failure)
    all_distances = distances(data)
    if __max_average_ball__(0, all_distances, goal_number) + laplace(0, 4/eps, 1) >\
                            goal_number - 2*promise - 4/eps*log(2/failure):
        return 0

    extended_domain = 2 ** int(ceil(log2(domain)))
    max_averages_by_radius = [__max_average_ball__(r, all_distances, goal_number)
                              for r in arange(0, extended_domain, 0.5)]

    def quality(d, r):
        try:
            return min(goal_number - max_averages_by_radius[r],
                   max_averages_by_radius[2*r] - goal_number + 4*promise) / 2
        except IndexError:
            raise IndexError('error while trying to qualify %f' % r)

    # TODO must complete those two
    def radius_interval_bounding(data_set, domain_end, j):
        return max(min(quality(data_set, i) for i in xrange(a, a + 2**j)) for a in xrange(domain_end - 2**j))

    def max_radius_in_interval(data_set, i):
        return max(quality(data_set, r) for r in i)

    return evaluate(data, domain, quality, promise,
                    0.5, eps, delta,
                    radius_interval_bounding, max_radius_in_interval)

