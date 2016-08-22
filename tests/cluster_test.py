import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as dss
from scipy.spatial.distance import euclidean
import src.cluster as cluster
import time
from mpl_toolkits.mplot3d import Axes3D
from __non_private_cluster__ import find_cluster


sample_number, dimension = 10000, 2
blobs = dss.make_blobs(sample_number, dimension, cluster_std=70)
blob = np.round(blobs[0], 2)

approximation, failure, eps, delta = 0.1, 0.1, 0.5, 2**-10
domain_end = max(abs(np.min(blob)), np.max(blob))
domain = (domain_end, 0.01)
desired_amount_of_points = 500

start_time = time.time()
radius, center = cluster.find(blob, dimension, domain, desired_amount_of_points,
                              approximation, failure, eps, delta, use_histograms=True)

end_time = time.time()
ball = [p for p in blob if euclidean(p, center) <= radius]
# blob = [p for p in blob if tuple(p) not in map(tuple, ball)]

fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(211, aspect='equal')
zipped_data = zip(*blob)
ax.scatter(*zipped_data, c='g')
zipped_ball = zip(*ball)
ax.scatter(*zipped_ball, c='r')

print "Good-radius: %d" % radius
print "Good-center: %s" % str(center)
print "Desired number of points in resulting ball: %d" % desired_amount_of_points
print "Number of points in the resulting ball: %d" % len(ball)
print "Run-time: %.2f seconds" % (end_time - start_time)

start_time = time.time()
test_radius, test_center = find_cluster(blob, desired_amount_of_points)
end_time = time.time()
ball = [p for p in blob if euclidean(p, test_center) <= test_radius]
zipped_data = zip(*blob)
ax2 = fig.add_subplot(212, aspect='equal')
ax2.scatter(*zipped_data, c='b')
zipped_ball = zip(*ball)
ax2.scatter(*zipped_ball, c='r')

print "Test-radius: %d" % test_radius
print "Test-center: %s" % str(test_center)
print "Desired number of points in resulting ball: %d" % desired_amount_of_points
print "Number of points in the resulting ball: %d" % len(ball)
print "Run-time: %.2f seconds" % (end_time - start_time)

plt.show()

