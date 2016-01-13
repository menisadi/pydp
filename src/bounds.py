from math import log, sqrt


def log_n(x, n):
    """
    recursive log
    preform natural log - n times
    :param x: input wchich the recursive log will be calculated upon
    :param n: number of times the log will be performed
    :return: ln(ln(ln...[n times]...ln(x)))...)
    """
    if n == 0:
        return x
    else:
        return log(log_n(x, n-1))


def step6_n2_bound(max_range, eps, alpha, beta):
    """
    for the case when the recursion bound N=2
    calculate the minimum sample size for which the call in step 6 will
    fail to return a 'good interval' only in probability < beta
    assuming the median problem so: r = samples/2

    :param max_range: maximum output possible
    :param eps: privacy parameter
    :param alpha: approximation parameter (from 0 to 1)
    :param beta: failre probability
    :return: the minimum samples required for step 6 to succeed
    """

    r = 16 * log(log(max_range, 2) / beta) / alpha / eps
    return 2*r


def dist_bound(eps, delta, alpha, beta):
    """
    calculate the minimum sample size for which A_dist at step 9 will fail
    only in probability < beta
    assuming the median problem so: r = samples/2

    :param eps, delta: privacy parameter
    :param alpha: approximation parameter (from 0 to 1)
    :param beta: failre probability
    :return: the minimum samples required for A_dist to run
    """

    r = 8 * log(1 / (beta * delta)) / alpha / eps / 3
    return 2*r


def rec_bound(t, rec, eps, delta, alpha, beta):
    """
    calculate the minimum sample size for which rec_concave will fail
    only in probability < beta
    taken from : # A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
    Theorem ----
    :return: the minimum samples required for rec_concave to run
    """
    return 8**rec*4/alpha/eps*(log(32/beta/delta) + log_n(t, rec))


def exponential_upper_bound(delta, beta):
    """
    calculate the maximum domain size for which the exponential mechanism
    will fail to return the highest rated element only in probability < beta
    assuming that the gap between the highest score and the second highest score is at least
    log (1 / (delta * beta)) / eps
    :return: maximum domain size
    """
    return float(sqrt(beta/delta))


def good_center_points_in_cluster(data_size, dimension, eps, delta, beta):
    """
    calculate the minimum amount of points required to be located in a cluster
    so algorithm good_center will return the center of the cluster in good probability
    :param data_size:
    :param dimension:
    :param eps:
    :param delta:
    :param beta:
    :return:
    """
    return 86645 * sqrt(dimension) / eps * log(64 * data_size * dimension / beta / eps / delta) * sqrt(log(8 / delta))


def choosing_mechanism_data_size(growth_bound, alpha, beta, eps, delta):
    """
    calculate the minimum data size in which choosing_mechanism will return a good approximation to the optimal result
    while preserving privacy
    :param growth_bound:
    :param alpha:
    :param beta:
    :param eps:
    :param delta:
    :return:
    """
    return 16 * log(16 * growth_bound / alpha / beta / eps / delta) / alpha / eps


