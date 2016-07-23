from src.good_radius import *
from numpy.random import normal
import numpy as np
import time
from numpy import array
from __non_private_cluster__ import find_cluster


sample_number, k, r = 2**12, 2, 4
center = 100
data_2d = array([int(i) for i in normal(center, 50, sample_number)]).reshape((sample_number/2, 2))
domain, goal_number, failure, eps, delta, promise = (0, 200), 2000, 0.1, 0.5, 2**-20, 300

r, c = find_cluster(data_2d, goal_number)
print "non-private radius : %d " % r
start_time = time.time()
result = find(data_2d, domain, goal_number, failure, eps)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = len([i for i in data_2d if np.linalg.norm(i - (center, center)) < result])
print "number of points in the resulting ball : %d" % points_in_resulting_ball

