import sklearn.datasets as dss
from scipy.spatial.distance import euclidean
import src.cluster as cluster
import time
from __non_private_cluster__ import *
from src.bounds import good_center_step_8_choosing_mechanism as bound


sample_number = 2000
dimension, domain = 100, (0, 1000)
blobs = dss.make_blobs(sample_number, dimension, cluster_std=50)
blob = blobs[0]

desired_amount_of_points, approximation, failure, eps, delta, promise = 1000, 0.1, 0.1, 0.5, 2**-10, 70
failure_bound = bound(sample_number, dimension, eps, delta, approximation)
print "The probability of failure is somewhere between %s" % str(failure_bound)
"""
start_time = time.time()
test_radius, test_center = find_cluster(blob, desired_amount_of_points)
middle_time = time.time()
test_ball = len([p for p in blob if euclidean(p, test_center) <= test_radius])
print "Test-radius: %d" % test_radius
# print "Test-center: %s" % str(test_center)
print "Number of points in the resulting ball: %d" % test_ball
print "Run-time: %.2f seconds" % (middle_time - start_time)
"""
for i in xrange(8):
    middle_time = time.time()
    try:
        radius, center = cluster.find(blob, dimension, domain, desired_amount_of_points,
                                      approximation, failure, eps, delta, shrink=False, use_filter=False)
        ball = [p for p in blob if euclidean(p, center) <= radius]
        print "Good-radius: %d" % radius
        # print "Good-center: %s" % str(center)
        print "Desired number of points in resulting ball: %d" % desired_amount_of_points
        print "Number of points in the resulting ball: %d" % len(ball)
    except ValueError:
        ball = []
        print '_|_'

    end_time = time.time()
    print "Run-time: %.2f seconds" % (end_time - middle_time)


