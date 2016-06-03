from src.good_center import find
from numpy import array
from numpy.random import normal
import numpy as np
import matplotlib.pyplot as plt
import time


"""
sample_number, k, r = 2**18, 2, 1
data_2d = np.random.normal(0, 3000, (sample_number, 2))
artificial_cluster_size = 2**14
artificial_cluster = np.random.normal(1000, 100, (artificial_cluster_size, 2))
data_2d = np.vstack((data_2d, artificial_cluster))
sample_number += artificial_cluster_size
"""

sample_number, k, r = 2**12, 2, 4
center = 100
data_2d = array([int(i) for i in normal(center, 50, sample_number)]).reshape((sample_number/2, 2))

start_time = time.time()
result = find(data_2d, sample_number, 2, 1, 20, 0.1, 0.05, 0.5, 2**-20)

print result

print "run-time: %.2f seconds" % (time.time() - start_time)


"""
older version
# result = best_box, box_quality(data, best_box), center_box, chosen_ball
ball = result[3]
ball = map(list, ball)
# print len(ball)


zipped_data = map(list, zip(*data_2d))
plt.scatter(*zipped_data, c='b')
zipped_ball = map(list, zip(*ball))
plt.scatter(*zipped_ball, c='r')
plt.show()
"""


