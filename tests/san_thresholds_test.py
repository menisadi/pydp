from __future__ import division
from numpy.random import exponential
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


start_time = time.time()

a, b, e, d = 0.05, 0.1, 0.5, 2**-20
samples_no = 1000
parameter = 5
data = [int(i) for i in exponential(parameter, samples_no)]
print len(data)
max_sample = max(data)
dim = ceil(log2(max_sample + 1))
end_domain = 2**int(dim)
print max_sample
san = sanitize(data, (0, end_domain), a, b, e, d)
print type(san)
