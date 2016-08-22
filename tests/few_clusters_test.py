from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import src.cluster as cluster
import time
from __non_private_cluster__ import *


samples = [1500, 1500, 1500]
desired_amount_of_points = 1000
means = [[500, 45000], [4, 30], [45000, 500]]
covs = [np.eye(2)*15, np.eye(2)*210, np.eye(2)*350]
clusters = [np.random.multivariate_normal(m, v, s) for m, v, s in zip(means, covs, samples)]
data = np.round(np.concatenate(([v for v in clusters])), 1)
domain_end = max(abs(np.min(data)), np.max(data))

dimension, domain = 2, (domain_end, 0.1)
approximation, failure, eps, delta = 0.1, 0.1, 0.5, 2**-10

for c in clusters:
    test_radius, test_center = find_cluster(np.round(c, 1), desired_amount_of_points)
    print "Radius: %d" % test_radius
    print "Center: %s" % str(test_center)

start_time = time.time()
test_radius, test_center = find_cluster(data, desired_amount_of_points)
print "Test-radius: %d" % test_radius
print "Test-center: %s" % str(test_center)
middle_time = time.time()
print "Run-time: %.2f seconds\n" % (middle_time - start_time)

radius, center = cluster.find(data, dimension, domain, desired_amount_of_points,
                              approximation, failure, eps, delta)

print "Good-radius: %d" % radius
print "Good-center: %s" % str(center)
end_time = time.time()
print "Run-time: %.2f seconds" % (end_time - middle_time)

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
ax.scatter(*zip(*data))
ax.annotate('center', xy=tuple(center), xytext=tuple(np.array(center)+100),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
phis = np.arange(0, 6.283, 0.01)
ax.plot(*circle(test_radius, phis, test_center), c='g', ls='-')
ax.plot(*circle(radius, phis, center), c='r', ls='-')
ball = sum(1 for p in data if norm(p-center) <= radius)
print "Number of points in the resulting ball: %d" % ball
plt.show()

