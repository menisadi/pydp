from src.good_radius import *
from numpy.random import normal
import numpy as np
import time
from numpy import round
from __non_private_cluster__ import find_cluster


sample_number = 2**13
center = 100
data_2d = round(normal(center, 50, (sample_number, 2)))
domain_end = max(abs(np.min(data_2d)), np.max(data_2d))
domain, goal_number = (domain_end, 1), 2000
failure, eps, delta = 0.1, 0.5, 2**-20

r, c = find_cluster(data_2d, goal_number)
print "non-private radius : %d " % r
start_time = time.time()
result = find(data_2d, domain, goal_number, failure, eps)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = len([i for i in data_2d if np.linalg.norm(i - (center, center)) <= result])
print "number of points in the resulting ball : %d" % points_in_resulting_ball

