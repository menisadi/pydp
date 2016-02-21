from collections import Counter
from numpy import log, log2, sqrt, abs, ceil
from numpy.random import randint, exponential
from src.bounds import choosing_mechanism_data_size
from src.san_points import sanitize
from src.qualities import concept_query
from src.examples import point_concept
import time


start_time = time.time()

a, b, e, d = 0.05, 0.1, 0.5, 2**-20
new_a = a / 2
new_b = a * b / 4
new_e = e / sqrt(32 * log(5/d) / a)
new_d = a * d / 5
samples_no = 100000  # int(choosing_mechanism_data_size(1, new_a, new_b, new_e, new_d)) + 1
parameter = 5
print samples_no
# data = [randint(1, parameter/2) for i in xrange(samples_no/5)]
# data.extend([randint(parameter/2, parameter) for k in xrange(4*samples_no/5+1)])
data = [int(i) for i in exponential(parameter, samples_no)]
print len(data)
max_sample = max(data)
dim = ceil(log2(max_sample + 1))
end_domain = 2**int(dim)
print max_sample
print dim
print Counter(data)
san = sanitize(data, a, b, e, d)
print [(z, concept_query(data, point_concept(z))) for z in xrange(end_domain)]
print [(z, san[z]) for z in xrange(end_domain)]
diffs = [abs(concept_query(data, point_concept(z)) - san[z]) for z in xrange(end_domain)]
print sum(i <= a for i in diffs)/float(end_domain)
print max(diffs)

print "run-time: %.2f seconds" % (time.time() - start_time)
