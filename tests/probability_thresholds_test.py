from __future__ import division
from numpy.random import exponential, normal, uniform
import time
from numpy import ceil, log2, linspace, mean, searchsorted
import matplotlib.pyplot as plt
from src.san_thresholds import sanitize
from statsmodels.distributions.empirical_distribution import ECDF


def plot_cdf(data_set):
    cf = ECDF(data_set)
    xs = linspace(0, max(data_set), 100)
    ys = [cf(j) for j in xs]
    plt.plot(xs, ys)


def plot_san_and_original(data_set, sanitized):
    sorted_san = sorted(sanitized)
    max_sample = max(data_set)
    i_max_san = searchsorted(sorted_san, max_sample)
    limited_san = sorted_san[:i_max_san]
    plot_cdf(data_set)
    plot_cdf(limited_san)
    plt.show()


def cdf_comp(data1, data2):
    """
    compare the cdf of two data-sets
    :param data1:
    :param data2:
    :return: the proportion of the data-sets which they differ by at least 'a'
    """
    m = int(ceil(max(max(data1), max(data2))))
    f1 = ECDF(data1)
    f2 = ECDF(data2)
    return 1-sum(1 for c in xrange(m) if abs(f1(c) - f2(c)) <= a)/m


def check(samples_size, alpha, beta, eps, delta, parameter):
    data = [int(i) for i in normal(0, parameter, samples_size)]
    m = min(data)
    data = [i-m for i in data]
    max_sample = max(data)
    dim = ceil(log2(max_sample + 1))
    end_domain = 2**int(dim)
    try:
        san = sanitize(data, (0, end_domain), alpha, beta, eps, delta)
        result = cdf_comp(san, data)
        if result == 0:
            plot_san_and_original(data, san)
    except ValueError:
        result = -1
    return result

start_time = time.time()

a, b, e, d = 0.1, 0.1, 0.5, 2**-20
b *= a / 231
p = 5
samples = 5000

iters = 10
checks = []
for i in xrange(iters):
    print i
    checks.append(check(samples, a, b, e, d, p))

not_bottom_results = [i for i in checks if i != -1]
didnt_get_bottom_prop = len(not_bottom_results)/iters
all_better_than_alpha = [i for i in not_bottom_results if i == 0]
perfect_result_prop = len(all_better_than_alpha)/iters

print "proportion of times we got perfect result: %.3f" % perfect_result_prop
print "proportion of times we didn't get 'bottom': %.3f" % didnt_get_bottom_prop
print "average within the non-bottom results: %.5f" % mean(not_bottom_results)

print "run-time: %.2f seconds" % (time.time() - start_time)
