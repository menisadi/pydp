from collections import Counter
from numpy import log, sqrt, abs
from numpy.random import randint, exponential
from src.bounds import choosing_mechanism_data_size
from src.san_points import sanitize
from src.qualities import concept_query
from src.examples import point_concept
import time


start_time = time.time()

a, b, e, d = 0.1, 0.1, 0.5, 2**-20
new_a = a / 2
new_b = a * b / 4
new_e = e / sqrt(32 * log(5/d) / a)
new_d = a * d / 5
samples_no = int(choosing_mechanism_data_size(1, new_a, new_b, new_e, new_d)) + 1
parameter = 20
print samples_no
data = [randint(1, parameter/2) for i in xrange(samples_no/5)]
data.extend([randint(parameter/2, parameter) for k in xrange(4*samples_no/5+1)])
# data = [int(i) for i in exponential(r, m)]
max_data = max(data)
print max_data
print len(data)
print Counter(data)
san = sanitize(data, set(range(max_data + 1)), a, b, e, d)
print [(z, concept_query(data, point_concept(z))) for z in xrange(1, max_data + 1)]
print san
print sum([abs(concept_query(data, point_concept(z)) - san[z]) <= a for z in xrange(1, max_data)])/float(max_data)

print "run-time: %.2f seconds" % (time.time() - start_time)
