from __future__ import division
from numpy import array, vstack, round
from numpy.random import normal, randn
import time
from numpy.linalg import norm
from src.good_radius import find
from __non_private_cluster__ import find_cluster
import numpy as np


sample_number, k, r = 2**12, 2, 4
center = 100
data_2d = round(normal(center, 1, (sample_number, 2)))
domain_end = max(abs(np.min(data_2d)), np.max(data_2d))
domain = domain_end, 1
goal_number, failure, eps, delta, promise = 1000, 0.1, 0.5, 2**-20, 300
point = randn(1, 2)
singular = array([point[0]]*1000)
data = vstack((data_2d, singular))

r, c = find_cluster(data_2d, goal_number)
print "non-private radius : %d " % r
start_time = time.time()
result = find(data, domain, goal_number, failure, eps)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = len([i for i in data if norm(i - point) <= result])
print "number of points in the resulting ball : %d" % points_in_resulting_ball
