from __future__ import division
from numpy.random import exponential, normal, uniform
import time
from numpy import ceil, log2, linspace
import matplotlib.pyplot as plt
from src.san_thresholds import sanitize


def cdf(data_set, x):
    return sum(1 for k in data_set if k < x)/len(data_set)


def plot_cdf(data_set, show=False):
    xs = linspace(0, end_domain, 100)
    ys = [cdf(data_set, j) for j in xs]
    plt.plot(xs, ys)
    if show:
        plt.show()


def cdf_comp(data1, data2):
    m = int(ceil(max(max(data1), max(data2))))
    return 1-sum(1 for c in xrange(m) if abs(cdf(data1, c) - cdf(data2, c)) <= a)/m


start_time = time.time()

a, b, e, d = 0.1, 0.1, 0.5, 2**-20
samples_no = 1000
parameter = 5
data = [int(i) for i in exponential(parameter, samples_no)]
m = min(data)
data = [i-m for i in data]
print len(data)
max_sample = max(data)
print max_sample
dim = ceil(log2(max_sample + 1))
print dim
end_domain = 2**int(dim)
# plot_cdf(data, True)
san = sanitize(data, (0, end_domain), a, b, e, d)
print cdf_comp(san, data)
plot_cdf(data)
plot_cdf(san)
plt.show()
