from math import log, sqrt, ceil, exp


def log_star(x):
    if x <= 1:
        return 0
    else:
        return 1 + log_star(log(x))


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

    :param eps, delta: privacy parameter
    :param alpha: approximation parameter (from 0 to 1)
    :param beta: failre probability
    :return: the minimum promise-parameter required for A_dist to run
    """

    r = 8 * log(1 / (beta * delta)) / alpha / eps / 3
    return r


def rec_bound(t, rec, eps, delta, alpha, beta):
    """
    calculate the minimum sample size for which rec_concave will fail
    only in probability < beta
    taken from : # A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
    Theorem ----
    :return: the minimum samples required for rec_concave to run
    """
    return 8**rec*4/alpha/eps*(log(32/beta/delta) + log_n(t, rec))


def exponential_bound(eps, alpha, beta, domain_size):
    """
    calculate the maximum domain size for which the exponential mechanism
    will fail to return the highest rated element only in probability < beta
    assuming that the gap between the highest score and the second highest score is at least
    log (1 / (delta * beta)) / eps
    :return: the minimum promise-parameter required for the exponential-mechanism to run well
    """
    r = 16 * log(log(domain_size, 2) / beta) / alpha / eps / 3
    return r


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


def san_points(eps, delta, alpha, beta):
    return 16/alpha**1.5/eps*sqrt(log(5/delta))*log(16/alpha/beta/eps/delta)


def histograms_bound(d, eps, delta):
    eps_tag = eps / sqrt(d*log(8/delta)) / 10.
    delta_tag = delta / float(d) / 8.
    return ceil(2 * log(2/delta_tag) / eps_tag)


def good_center_step_8_choosing_mechanism(s, d, eps, delta, alpha):
    """
    lower bound the probability of failure
    :return:
    """
    def bound_by_best_score(m):
        return exp(s*(alpha - 2*m) * eps / 80. / sqrt(d*log(8/delta))) / 2.
    return bound_by_best_score(1), bound_by_best_score(0)

