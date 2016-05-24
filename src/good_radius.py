from __future__ import division
from numpy import zeros, sum, sqrt, log
from numpy.linalg import norm
import numpy as np
from numpy.random import laplace
from bounds import log_star
from basicdp import exponential_mechanism


def __neighbours__(points, radius):
    n = len(points)
    neighbourhood = zeros((n, n))
    for i in xrange(n):
        for j in xrange(i):
            if norm(points[i]-points[j]) <= radius:
                neighbourhood[i, j] = neighbourhood[j, i] = 1
    return neighbourhood


def __max_average_ball__(points, radius, t):
    hood = __neighbours__(points, radius)
    a = sum(hood, axis=1).argsort()[t:]
    return sum(min(sum(i), t) for i in hood[a]) / t


def __promise__(data, domain, eps, delta, failure):
    const = log_star(2 * (domain[1] - domain[0] + 1) * sqrt(data.shape[1]))
    return 8 ** const * 144 * const / eps * log(24 * const / failure / delta)


# the parameter promise should later be removed from the input and be calculated within the function
def find(data, domain, goal_number, failure, eps, delta, promise):
    # domain = min(np.min(data, axis=0)), max(np.max(data, axis=0))
    if __max_average_ball__(data, 0, goal_number) + laplace(0, 4/eps, 1) >\
                            goal_number - 2*promise - 4/eps*log(2/failure):
        return 0

    def quality(d, r):
        return min(goal_number - __max_average_ball__(d, r/2, goal_number),
                   __max_average_ball__(d, r, goal_number) - goal_number + 4*promise) / 2

    return exponential_mechanism(data, range(domain[1]-domain[0]), quality, eps)

