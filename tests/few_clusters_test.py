from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import src.cluster as cluster
import time
from __non_private_cluster__ import *


sample_number, desired_amount_of_points = 2000, 1800

means = [[500, 30000], [4, 30], [30000, 500]]
covs = [np.eye(2)*5, np.eye(2)*20, np.eye(2)*100]
clusters = [np.random.multivariate_normal(m, v, sample_number) for m, v in zip(means, covs)]
data = np.round(np.concatenate(([v for v in clusters])), 1)
domain_end = max(abs(np.min(data)), np.max(data))

dimension, domain = 2, (domain_end, 0.1)
approximation, failure, eps, delta, promise = 0.1, 0.1, 0.5, 2**-10, 200

start_time = time.time()
test_radius, test_center = find_cluster(data, desired_amount_of_points)
print test_radius
middle_time = time.time()
print "Run-time: %.2f seconds" % (middle_time - start_time)

radius, center = cluster.find(data, dimension, domain, desired_amount_of_points,
                              approximation, failure, eps, delta)
print radius
print center
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
print sum(1 for p in data if norm(p-center) <= radius)
plt.show()

