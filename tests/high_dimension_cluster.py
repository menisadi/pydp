import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dss
from scipy.spatial.distance import euclidean
import src.cluster as cluster
import time


sample_number = 100
dimension, domain = 800, (0, 70)
blobs = dss.make_blobs(sample_number, dimension, cluster_std=70)
blob = blobs[0]

desired_amount_of_points, approximation, failure, eps, delta, promise = 500, 0.1, 0.1, 0.5, 2**-10, 70

start_time = time.time()
radius, center = cluster.find(blob, dimension, domain, desired_amount_of_points,
                              approximation, failure, eps, delta, promise, shrink=True)

ball = [p for p in blob if euclidean(p, center) <= radius]
end_time = time.time()

blob = [p for p in blob if tuple(p) not in map(tuple, ball)]

print "Good-radius: %d" % radius
print "Good-center: %s" % str(center)
print "Desired number of points in resulting ball: %d" % desired_amount_of_points
print "Number of points in the resulting ball: %d" % len(ball)
print "Run-time: %.2f seconds" % (end_time - start_time)

plt.show()



