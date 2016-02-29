from numpy.random import laplace
from math import log, ceil


def points_in_subset(data, subset):
    return len([i for i in data if subset[0] <= i <= subset[1]])


def sanitize(samples, domain_range, alpha, beta, eps, delta):
    san_data = []
    calls = 77 / alpha
    return rec_sanitize(samples, domain_range, alpha, beta, eps, delta, san_data, calls)


def rec_sanitize(samples, domain_range, alpha, beta, eps, delta, san_data, calls):
    # step 1
    if calls == 0:
        return
    calls -= 1
    # step 2
    noisy_points_in_range = points_in_subset(samples, domain_range) + laplace(0, 1/eps, 1)
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
    # step 5
    for j in xrange(log_size+1):
        pass
    return

