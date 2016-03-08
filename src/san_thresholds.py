from numpy.random import laplace
from math import log, ceil
from qualities import points_in_subset, point_count_intervals_bounding
from basicdp import exponential_mechanism, choosing_mechanism
from bounds import log_star
from functools import partial


def sanitize(samples, domain_range, alpha, beta, eps, delta):
    san_data = []
    calls = 77 / alpha
    return __rec_sanitize__(samples, domain_range, alpha, beta, eps, delta, san_data, calls)


def __rec_sanitize__(samples, domain_range, alpha, beta, eps, delta, san_data, calls):
    # step 1
    if calls == 0:
        return
    calls -= 1

    # step 2
    samples_domain_points = points_in_subset(data = samples)
    noisy_points_in_range = samples_domain_points(domain_range) + laplace(0, 1/eps, 1)
    sample_size = len(samples)

    # step 3
    if noisy_points_in_range < alpha*sample_size/8:
        base_range = domain_range
        san_data.extend(base_range[1] * noisy_points_in_range)
        return san_data

    # step 4
    domain_size = domain_range[1] - domain_range[0] + 1
    log_size = int(ceil(log(domain_size, 2)))
    size_tag = 2**log_size

    # step 6

    def quality(data, j):
        min(point_count_intervals_bounding(data, domain_range, j)-alpha * sample_size / 32,
            3 * alpha * sample_size / 32 - point_count_intervals_bounding(data, domain_range, j-1))

    # step 7
    promise = alpha * sample_size / 32

    # step 8
    # TODO check if it true that : d == log_size
    new_eps = eps/3/log_star(log_size)
    new_delta = delta/3/log_star(log_size)
    # TODO replace with rec_concave
    z_tag = exponential_mechanism(samples, range(log_size+1), quality, new_eps, new_delta)
    z = 2 ** z_tag

    #step 9
    if z == 0:
        b = choosing_mechanism(samples, range(domain_range[0], domain_range[1] + 1), samples_domain_points,
                               1, alpha/64., beta, eps, delta)
        a = b
    else:
        pass

    return

