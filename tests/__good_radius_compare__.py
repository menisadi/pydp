# return here after good_radius_concave is complete
from src.good_radius import find
from src.good_radius_concave import find as find2
import numpy as np
import time
from __non_private_cluster__ import find_cluster
import sklearn.datasets as dss


sample_number = 3000
dimension, domain = 3, (0, 100)
blobs = dss.make_blobs(sample_number, dimension, cluster_std=70)
blob = blobs[0]

desired_amount_of_points, approximation, failure, eps, delta, promise = 1000, 0.1, 0.1, 0.5, 2**-10, 70

r, c = find_cluster(blob, desired_amount_of_points)
print r

start_time = time.time()
result = find(blob, domain, desired_amount_of_points, failure, eps)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = len([i for i in blob if np.linalg.norm(i - (c, c)) < result])
print "number of points in the resulting ball : %d" % points_in_resulting_ball

start_time = time.time()
result = find2(blob, domain, desired_amount_of_points, failure, eps, delta, promise)
print "run-time: %.2f seconds" % (time.time() - start_time)
print "good radius : %d " % result
points_in_resulting_ball = len([i for i in blob if np.linalg.norm(i - (c, c)) < result])
print "number of points in the resulting ball : %d" % points_in_resulting_ball


