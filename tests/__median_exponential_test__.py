"""
An atempt to compare the result from flat_concave and the basic exponential-mechanism
 right now - not working due to memory problem
"""
from src.basicdp import exponential_mechanism
import src.examples
import numpy as np
import src.bounds
import time
from src.qualities import bulk_quality_minmax, quality_minmax, min_max_maximum_quality


def check(t, alpha, eps, samples_size):
    range_end = 2**t
    data_center = np.random.uniform(range_end/3, range_end/3*2)
    data = src.examples.get_random_data(samples_size, pivot=data_center)
    # data = examples.get_random_data(samples_size, 'bimodal')
    data = sorted(filter(lambda x: 0 <= x <= range_end, data))
    maximum_quality = min_max_maximum_quality(data, (0, range_end))
    quality_result_lower_bound = maximum_quality * (1-alpha)
    try:
        result = exponential_mechanism(data, range(range_end), bulk_quality_minmax, eps, True)
        result_quality = quality_minmax(data, result)
    except ValueError:
        # result = -1
        result_quality = -1
    return result_quality != -1, result_quality >= quality_result_lower_bound, result_quality/float(maximum_quality)

start_time = time.time()

range_end_exponent = 30
my_alpha = 0.1
my_eps = 0.1
my_delta = 2**-30
# note that flat-concave preserve (4*eps,delta)-differential privacy
my_beta = 0.1

# here we can play with the sample size if we want
samples = 1200

iters = 10
checks = []
for i in xrange(iters):
    print i
    checks.append(check(range_end_exponent, my_alpha, my_eps, samples))

did_not_fail = sum(i[0] for i in checks)
good_quality = sum(i[1] for i in checks)
min_quality = min(i[2] for i in checks)
average_quality = np.average([i[2] for i in checks])
print "proportion of times Adist returned a value: %.2f" % (did_not_fail/float(iters))
print "proportion of times we got good quality: %.2f" % (good_quality/float(iters))
print "minimum max_quality/Q(result): %.2f" % min_quality
print "average max_quality/Q(result): %.2f" % average_quality
print "run-time: %.2f seconds" % (time.time() - start_time)
