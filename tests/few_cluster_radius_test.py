from src.good_radius import *
from numpy.random import randint, normal
import numpy as np
import time


sample_number, desired_amount_of_points = 2000, 500

means = [[50, 50], [4, 30], [75, 20]]
covs = [np.eye(2), np.eye(2)*7, np.eye(2)*20]
clusters = [np.random.multivariate_normal(m, v, sample_number) for m, v in zip(means, covs)]
data = np.concatenate(([v for v in clusters]))

start_time = time.time()
domain, goal_number, failure, eps, delta, promise = (0, 50), 2000, 0.1, 0.5, 2**-20, 300
result = find(data, domain, goal_number, failure, eps)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = [len([i for i in data if np.linalg.norm(i - m) < result]) for m in means]

print "number of points in the resulting balls : %s" % str(points_in_resulting_ball)

