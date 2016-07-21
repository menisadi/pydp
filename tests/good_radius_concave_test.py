from src.good_radius_concave import *
from numpy.random import normal
import numpy as np
import time
from numpy import array
from __non_private_cluster__ import find_cluster
import matplotlib.pyplot as plt


sample_number = 1000
center = 100
data_2d = array([int(i) for i in normal(center, 25, sample_number)]).reshape((sample_number/2, 2))
domain, goal_number, failure, eps, delta = 100, 500, 0.1, 0.5, 2**-20
# plt.scatter(*zip(*data_2d))
# plt.show()

promise = 100
# r, c = find_cluster(data_2d, goal_number)
# print "non-private radius : %d " % r
start_time = time.time()
result = find(data_2d, goal_number, failure, eps, delta, promise)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = len([i for i in data_2d if np.linalg.norm(i - (center, center)) < result])
print "number of points in the resulting ball : %d" % points_in_resulting_ball
# points_in_resulting_ball = len([i for i in data_2d if np.linalg.norm(i - (c, c)) < result])
# print "number of points in the resulting ball : %d" % points_in_resulting_ball
