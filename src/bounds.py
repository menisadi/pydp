import numpy as np


def log_n(x, n):
    if n == 0:
        return x
    else:
        return np.log(log_n(x, n-1))


def dist_bound(eps, delta, alpha, beta):
    """
    calculate the minimum sample size for which A_dist at step 9 will fail
    only in probability < beta
    assuming the median problem so: r = samples/2
    :return: the minimum samples required for A_dist to run
    """
    r = 8 * np.log(1 / (beta * delta)) / alpha / eps /2
    return 2*r


def rec_bound(t, rec, eps, delta, alpha, beta):
    return 8**rec*4/alpha/eps*(np.log(32/beta/delta) + log_n(t, rec))


def exponential_bound(delta, beta):
    return float(np.sqrt(beta/delta))


my_alpha = 0.2
my_beta = 0.01
my_rec = 2
my_eps = 0.5

exponent = 10
my_t = 2**exponent
my_delta = 1/float(my_t)
db = dist_bound(my_eps, my_delta, my_alpha, my_beta)
rb = rec_bound(my_t, my_rec, my_eps, my_delta, my_alpha, my_beta)
eb = exponential_bound(my_delta, my_beta)

print "n: %d" % exponent
print "T: %d" % my_t
print "dist bound: %.2f" % db
print "proportion to T: %.5f" % (db/my_t)
print "rec bound: %.2f" % rb
print "proportion to T: %.5f" % (rb/my_t)
print "exponential bound: %.2f" % eb
print "proportion to T: %.5f" % (eb/my_t)

