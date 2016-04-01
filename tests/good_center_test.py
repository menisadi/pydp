from src.good_center import find
import numpy as np
import matplotlib.pyplot as plt
import time


sample_number, k, r = 2**18, 2, 1
data_2d = np.random.normal(0, 3000, (sample_number, 2))
artificial_cluster_size = 2**14
artificial_cluster = np.random.normal(1000, 100, (artificial_cluster_size, 2))
data_2d = np.vstack((data_2d, artificial_cluster))
sample_number += artificial_cluster_size
start_time = time.time()
result = find(data_2d, sample_number, 2, 1, artificial_cluster_size, 0.1, 0.05, 0.5, 2**-20)
print result[:3]
ball = result[3]
ball = map(list, ball)
# print len(ball)
print "run-time: %.2f seconds" % (time.time() - start_time)

zipped_data = map(list, zip(*data_2d))
plt.scatter(*zipped_data, c='b')
zipped_ball = map(list, zip(*ball))
plt.scatter(*zipped_ball, c='r')
plt.show()

