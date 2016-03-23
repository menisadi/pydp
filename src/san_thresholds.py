from numpy.random import laplace
from math import log, ceil
from qualities import points_in_subset, point_count_intervals_bounding
from basicdp import exponential_mechanism, choosing_mechanism
from bounds import log_star
from functools import partial
from examples import __build_intervals_set__
from collections import Counter


calls = 0
san_data = []


def sanitize(samples, domain_range, alpha, beta, eps, delta):
    global calls
    calls = 77 / alpha
    domain_size = domain_range[1] - domain_range[0] + 1
    dim = int(ceil(log(domain_size, 2)))
    return __rec_sanitize__(samples, domain_range, alpha, beta, eps, delta, dim)


def __rec_sanitize__(samples, domain_range, alpha, beta, eps, delta, dimension):
    # print domain_range
    global calls
    global san_data
    # step 1
    if calls == 0:
        return
    calls -= 1

    # step 2
    # the use of partial is redundant
    samples_domain_points = partial(points_in_subset, samples)
    noisy_points_in_range = samples_domain_points(subset=domain_range) + laplace(0, 1/eps, 1)
    sample_size = len(samples)

    # step 3
    if noisy_points_in_range < alpha*sample_size/8:
        base_range = domain_range
        san_data.extend(base_range[1] * noisy_points_in_range)
        return san_data

    # step 4
    domain_size = domain_range[1] - domain_range[0] + 1
    log_size = int(ceil(log(domain_size, 2)))
    # not needed
    # size_tag = 2**log_size

    # step 6

    def quality(data, j):
        return min(point_count_intervals_bounding(data, domain_range, j)-alpha * sample_size / 32,
                3 * alpha * sample_size / 32 - point_count_intervals_bounding(data, domain_range, j-1))

    # not needed if using exponential_mechanism
    # step 7
    # promise = alpha * sample_size / 32

    # step 8
    new_eps = eps/3/log_star(dimension)
    # new_delta = delta/3/log_star(dimension)
    # note the use of exponential_mechanism instead of rec_concave
    z_tag = exponential_mechanism(samples, range(log_size+1), quality, new_eps)
    z = 2 ** z_tag

    # step 9
    if z_tag == 0:
        point_counter = Counter(samples)

        def special_quality(data, b):
            return point_counter[b]

        b = choosing_mechanism(samples, range(domain_range[0], domain_range[1] + 1), special_quality,
                               1, alpha/64., beta, eps, delta)
        a = b
    # step 10
    else:
        first_intervals = __build_intervals_set__(samples, 2*z, domain_range[0], domain_range[1] + 1)
        second_intervals = __build_intervals_set__(samples, 2*z_tag, domain_range[0], domain_range[1] + 1, True)
        intervals = [(i, i+2*z-1) for i in first_intervals+second_intervals]
        a, b = choosing_mechanism(samples, intervals, points_in_subset, 2, alpha/64., beta, eps, delta)

    if type(a) == str:
        raise ValueError("stability problem - choosing_mechanism returned 'bottom'")

    # step 11
    # although not mentioned I assume the noisy value should be rounded
    noisy_count_ab = int(samples_domain_points((a, b)) + laplace(0, 1/eps, 1))
    san_data.extend([b] * noisy_count_ab)

    # step 12
    if a > domain_range[0]:
        rec_range = (domain_range[0], a - 1)
        __rec_sanitize__(samples, rec_range, alpha, beta, eps, delta, dimension)
    if b < domain_range[1]:
        rec_range = (b + 1, domain_range[1])
        __rec_sanitize__(samples, rec_range, alpha, beta, eps, delta, dimension)
    return san_data
