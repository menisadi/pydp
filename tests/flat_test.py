import src.flat_concave
import src.examples
import numpy as np
import src.bounds
import src.qualities


range_end = 2**40

alpha = 0.1
eps = 0.1
delta = 1e-6
beta = 0.01

samples_size = 1000  # int(bounds.step6_n2_bound(range_end, eps, alpha, beta))
print "range size: %d" % range_end
print "sample size: %d" % samples_size
data_center = np.random.uniform(range_end/3, range_end/3*2)
data = src.examples.get_random_data(samples_size, pivot=data_center)
data = sorted(filter(lambda x: 0 <= x <= range_end, data))

maximum_quality = src.qualities.min_max_maximum_quality(data, (0, range_end))

print "the exact median is: %d" % np.median(data)
print "the best quality of a domain element: %d" % maximum_quality
quality_result_lower_bound = maximum_quality*(1-alpha)
print 'minimum "allowed" quality: %d' % quality_result_lower_bound

print "testing flat_concave to find median"

try:
    result = src.flat_concave.evaluate(data, range_end, src.qualities.quality_minmax, maximum_quality, alpha, eps, delta,
                                   src.qualities.min_max_intervals_bounding, src.qualities.min_max_maximum_quality,True)
    result_quality = src.qualities.quality_minmax(data, result)
except ValueError:
    print "Adist returned 'Bottom'"
    result = -1
    result_quality = -1

print "result from flat_concave: %d" % result
print "and its quality: %d \n" % result_quality
