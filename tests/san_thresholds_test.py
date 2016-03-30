from __future__ import division
from numpy.random import exponential, normal, chisquare, uniform
import time
from numpy import ceil, log2, linspace, searchsorted
import matplotlib.pyplot as plt
from src.san_thresholds import sanitize
from statsmodels.distributions.empirical_distribution import ECDF


def plot_cdf(data_set):
    cf = ECDF(data_set)
    xs = linspace(0, max(data_set), 100)
    ys = [cf(j) for j in xs]
    plt.plot(xs, ys)


def cdf_comp(data1, data2, alpha):
    """
    :param alpha:
    :param data1:
    :param data2:
    :return: percentage of of the cdf's which differ in more than alpha
    """
    m = int(ceil(max(max(data1), max(data2))))
    f1 = ECDF(data1)
    f2 = ECDF(data2)
    return 1-sum(1 for c in xrange(m) if abs(f1(c) - f2(c)) <= alpha)/m


a, b, e, d = 0.1, 0.1, 0.5, 2**-20
b *= a / 231
samples_no = 5000
parameter = 5
data = [int(i) for i in normal(0, parameter, samples_no)]
m = min(data)
data = [i-m for i in data]
print len(data)
max_sample = max(data)
print max_sample
dim = ceil(log2(max_sample + 1))
print dim
end_domain = 2**int(dim)
start_time = time.time()
san = sanitize(data, (0, end_domain), a, b, e, d)
run_time = time.time() - start_time
print max(san)
print cdf_comp(san, data, a)
sorted_san = sorted(san)
i_max_san = searchsorted(sorted_san, max_sample)
limited_san = sorted_san[:i_max_san]
plot_cdf(data)
plot_cdf(limited_san)
plt.show()

print "run-time: %.2f seconds" % run_time
