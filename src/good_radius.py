from __future__ import division
from numpy import sum, log, sqrt, ceil
from basicdp import exponential_mechanism_big
from sklearn.metrics.pairwise import euclidean_distances as distances
from numpy.random import laplace


def __max_average_ball__(radius, hood, t):
    # TODO docstring
    """

    :param radius:
    :param hood:
    :param t:
    :return:
    """
    close_points = (hood <= radius).astype(int)
    a = sum(close_points, axis=1).argsort()[-t:]
    return sum(min(sum(i), t) for i in close_points[a]) / t


def sparse_domain(domain, dimension):
    domain_end, domain_interval = domain
    domain_end *= 2*ceil(sqrt(dimension))
    new_domain = [domain_end]
    while domain_end >= domain_interval:
        domain_end /= 2
        new_domain.append(domain_end)
    return new_domain


def find(data, domain, goal_number, failure, eps):
    # TODO docstring
    """

    :param data:
    :param domain: (absolute value of domain's end as int, minimum intervals in domain as float)
    :param goal_number:
    :param failure:
    :param eps:
    :return:
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
    sparsed_domain = sparse_domain(domain, dimension)
    # max_averages_by_radius = [__max_average_ball__(r, all_distances, goal_number)
    #                          for r in sparsed_domain]

    def quality(d, r):
        return min(goal_number - __max_average_ball__(r / 2, all_distances, goal_number),
                   __max_average_ball__(r, all_distances, goal_number) - goal_number + 2*a) / 2

    return exponential_mechanism_big(data, sparsed_domain, quality, eps / 2)

