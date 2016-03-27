from __future__ import division
from numpy.random import exponential, normal, uniform
import time
from numpy import ceil, log2, linspace, mean
import matplotlib.pyplot as plt
from src.san_thresholds import sanitize
from statsmodels.distributions.empirical_distribution import ECDF


def plot_cdf(data_set, show=False):
    cf = ECDF(data_set)
    xs = linspace(0, max(data_set), 100)
    ys = [cf(j) for j in xs]
    plt.plot(xs, ys)
    if show:
        plt.show()


def cdf_comp(data1, data2):
    m = int(ceil(max(max(data1), max(data2))))
    f1 = ECDF(data1)
    f2 = ECDF(data2)
    return 1-sum(1 for c in xrange(m) if abs(f1(c) - f2(c)) <= a)/m


def check(samples_size, alpha, beta, eps, delta):
    data = [int(i) for i in exponential(parameter, samples_size)]
    m = min(data)
    data = [i-m for i in data]
    max_sample = max(data)
    dim = ceil(log2(max_sample + 1))
    end_domain = 2**int(dim)
    try:
        san = sanitize(data, (0, end_domain), alpha, beta, eps, delta)
        result = cdf_comp(san, data)
    except ValueError:
        result = -1
    return result

start_time = time.time()

a, b, e, d = 0.1, 0.1, 0.5, 2**-20
parameter = 20
samples = 10000

iters = 5
checks = []
for i in xrange(iters):
    print i
    checks.append(check(samples, a, b, e, d))

not_bottom_results = [i for i in checks if i != -1]
better_than_alpha_prop = sum(1 for i in checks if -1 < i <= a)/iters
didnt_get_bottom = sum(1 for i in checks if i != -1)/iters
# print checks
print "proportion of times we got good result: %.2f" % better_than_alpha_prop
print "proportion of times we didn't get good 'bottom': %.2f" % didnt_get_bottom
print "average within the non-bottom results: %.2f" % mean(not_bottom_results)
print "run-time: %.2f seconds" % (time.time() - start_time)
