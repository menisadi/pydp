from __future__ import division
from numpy import sum, log, sqrt, ceil, arange
import numpy as np
from basicdp import exponential_mechanism_big
from sklearn.metrics.pairwise import euclidean_distances as distances
from numpy.random import laplace


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


def __create_regular_domain__(domain, dimension):
    """
    Helper function. Used as input for the Exponential Mechanism at the last step of the procedure 'find'
    Given the domain properties and the dimension builds the domain of possible radius value
    :param domain: tuple(absolute value of domain's end as int, minimum intervals in domain as float)
    :param dimension: dimension of the vector space from which the data is drawn
    :return:
    """
    domain_end, domain_interval = domain
    domain_end *= 2*ceil(sqrt(dimension))
    return np.delete(arange(0, domain_end + domain_interval, domain_interval), 0)


def __sparse_domain__(domain, dimension):
    """
    Helper function. Used as input for the Exponential Mechanism at the last step of the procedure 'find'
    Given the domain properties and the dimension builds a sparse version of the possible radius values
    The list will not contain all the possible values but is build in a way which the Good-Radius
    guarantees for the qualities of the resulting radius is maintained.
    :param domain: tuple(absolute value of domain's end as int, minimum intervals in domain as float)
    :param dimension: dimension of the vector space from which the data is drawn
    :return:
    """
    domain_end, domain_interval = domain
    domain_end *= 2*ceil(sqrt(dimension))
    new_domain = [domain_end]
    while domain_end >= domain_interval:
        domain_end /= 2
        new_domain.append(domain_end)
    return new_domain


def find(data, domain, goal_number, failure, eps, sparse=True):
    """
    Based on "Locating a Small Cluster Privately" by Kobbi Nissim, Uri Stemmer, and Salil Vadhan. PODS 2016.
    Given a data set, finds the radius of an approximately minimal cluster of points with
    approximately the desired amount of points
    :param data: list of points in R^dimension
    :param domain: tuple(absolute value of domain's end as int, minimum intervals in domain as float)
    :param goal_number: the number of desired points in the resulting cluster
    :param failure: 0 < float < 1. chances that the procedure will fail to return an answer
    :param eps: float > 0. privacy parameter
    :param sparse: 1 > float > 0. privacy parameter
    :return: the radius of the resulting cluster
    """
    # max(abs(np.min(data)), np.max(data))
    all_distances = distances(data)
    # TODO change variable name
    # 'a' need to greater than - log(domain[0] / failure) / eps
    a = 2 * log(domain[0] / failure) / eps
    thresh = goal_number - a - log(1 / failure) / eps
    # TODO verify that the noise addition is correct
    if __max_average_ball__(0, all_distances, goal_number) + laplace(0, 1 / eps, 1) > thresh:
        return 0

    dimension = data.shape[1]
    # TODO maybe a little less sparse?
    if sparse:
        new_domain = __sparse_domain__(domain, dimension)
    else:
        new_domain = __create_regular_domain__(domain, dimension)

    def quality(d, r):
        return min(goal_number - __max_average_ball__(r / 2, all_distances, goal_number),
                   __max_average_ball__(r, all_distances, goal_number) - goal_number + 2*a) / 2

    return exponential_mechanism_big(data, new_domain, quality, eps / 2)

