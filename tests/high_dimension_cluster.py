import sklearn.datasets as dss
from scipy.spatial.distance import euclidean
import src.cluster as cluster
import time
from __non_private_cluster__ import *


sample_number = 1000
dimension, domain = 800, (0, 3000)
blobs = dss.make_blobs(sample_number, dimension, cluster_std=70)
blob = blobs[0]

desired_amount_of_points, approximation, failure, eps, delta, promise = 700, 0.1, 0.1, 0.5, 2**-10, 70

start_time = time.time()
test_radius, test_center = find_cluster(blob, desired_amount_of_points)
middle_time = time.time()
test_ball = len([p for p in blob if euclidean(p, test_center) <= test_radius])
print "Test-radius: %d" % test_radius
# print "Test-center: %s" % str(test_center)
print "Number of points in the resulting ball: %d" % test_ball
print "Run-time: %.2f seconds" % (middle_time - start_time)

radius, center = cluster.find(blob, dimension, domain, desired_amount_of_points,
                              approximation, failure, eps, delta, shrink=True, use_filter=True)

ball = [p for p in blob if euclidean(p, center) <= radius]
end_time = time.time()

blob = [p for p in blob if tuple(p) not in map(tuple, ball)]

print "Good-radius: %d" % radius
# print "Good-center: %s" % str(center)
print "Desired number of points in resulting ball: %d" % desired_amount_of_points
print "Number of points in the resulting ball: %d" % len(ball)
print "Run-time: %.2f seconds" % (end_time - middle_time)


