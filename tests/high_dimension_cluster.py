import sklearn.datasets as dss
from scipy.spatial.distance import euclidean
import src.cluster as cluster
import time
from __non_private_cluster__ import find_cluster
from src.bounds import good_center_step_8_choosing_mechanism as bound
import numpy as np


sample_number = 5000
dimension = 20
blobs = dss.make_blobs(sample_number, dimension, cluster_std=50)
blob = np.round(blobs[0], 2)

approximation, failure, eps, delta = 0.1, 0.1, 0.5, 2**-10
domain_end = max(abs(np.min(blob)), np.max(blob))
domain = (domain_end, 0.01)
desired_amount_of_points = 1000

failure_bound = bound(sample_number, dimension, eps, delta, approximation)
print "The probability of failure is somewhere between %s\n" % str(failure_bound)

radius, center = find_cluster(blob, desired_amount_of_points)
print "Test-radius: %d" % radius

for i in xrange(8):
    middle_time = time.time()
    try:
        radius, center = cluster.find(blob, dimension, domain, desired_amount_of_points,
                                      approximation, failure, eps, delta, shrink=False, use_histograms=False)
        ball = [p for p in blob if euclidean(p, center) <= radius]
        print "Good-radius: %d" % radius
        # print "Good-center: %s" % str(center)
        print "Desired number of points in resulting ball: %d" % desired_amount_of_points
        print "Number of points in the resulting ball: %d" % len(ball)
    except ValueError:
        ball = []
        print '_|_'

    end_time = time.time()
    print "Run-time: %.2f seconds\n" % (end_time - middle_time)


