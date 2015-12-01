from numpy import log, sqrt


def log_n(x, n):
    if n == 0:
        return x
    else:
        return log(log_n(x, n-1))


def step6_n2_bound(max_range, eps, alpha, beta):
    """
    for the case when the recurssion bound N=2
    calculate the minimum sample size for which the call in step 6 will
    fail to return a 'good interval' only in probability < beta
    assuming the median problem so: r = samples/2
    :return: the minimum samples required for step 6 to succeed
    """
    r = 16 * log(log(max_range) / beta) / alpha / eps
    return 2*r

def dist_bound(eps, delta, alpha, beta):
    """
    calculate the minimum sample size for which A_dist at step 9 will fail
    only in probability < beta
    assuming the median problem so: r = samples/2
    :return: the minimum samples required for A_dist to run
    """
    r = 8 * log(1 / (beta * delta)) / alpha / eps / 3
    return 2*r


def rec_bound(t, rec, eps, delta, alpha, beta):
    """
    calculate the minimum sample size for which rec_concave will fail
    only in probability < beta
    taken from : # A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
    Theorem ----
    :return: the minimum samples required for rec_concave to run
    """
    return 8**rec*4/alpha/eps*(log(32/beta/delta) + log_n(t, rec))


def exponential_upper_bound(delta, beta):
    """
    calculate the maximum domain size for which the exponential mechanism
    will fail to return the highest rated element only in probability < beta
    assuming that the gap between the highest score and the second highest score is at least
    log (1 / (delta * beta)) / eps
    :return: maximum domain size
    """
    return float(sqrt(beta/delta))


"""
my_alpha = 0.2
my_beta = 0.01
my_rec = 2
my_eps = 0.5

exponent = 14
my_t = 2**exponent
my_delta = 1/float(my_t)
db = dist_bound(my_eps, my_delta, my_alpha, my_beta)
rb = rec_bound(my_t, my_rec, my_eps, my_delta, my_alpha, my_beta)
eb = exponential_upper_bound(my_delta**3, my_beta)  # for this one we must take a very small delta

print "n: %d" % exponent
print "T: %d\n" % my_t
print "dist bound: %.2f" % db
print "proportion to T: %.5f" % (db/my_t)
print "rec bound: %.2f" % rb
print "proportion to T: %.5f" % (rb/my_t)
print "\nexponential upper bound on the domain: %.2f" % eb
"""