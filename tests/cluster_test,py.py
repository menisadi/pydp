import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dss
from scipy.spatial.distance import euclidean
import src.cluster as cluster


sample_number = 5000
blobs = dss.make_blobs(sample_number, cluster_std=500)
blob = blobs[0]
plt.scatter(*zip(*blob))
plt.show()

dimension, domain = 2, (0, 500)
desired_amount_of_points, approximation, failure, eps, delta, promise = 1000, 0.1, 0.1, 0.5, 2**-10, 100

radius, center = cluster.find(blob, dimension, domain, desired_amount_of_points,
                              approximation, failure, eps, delta, promise)

ball = [p for p in blob if euclidean(p, center) <= radius]
print "number of points in the resulting ball: %d" % len(ball)

zipped_data = zip(*blob)
plt.scatter(*zipped_data, c='b')
zipped_ball = zip(*ball)
plt.scatter(*zipped_ball, c='r')

print "the radius: %d" % radius
print "the center: %s" % str(center)
print "number of points in the resulting ball: %d" % len(ball)






