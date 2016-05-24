from src.good_radius import *
from numpy.random import randint
import numpy as np
import matplotlib.pyplot as plt
import time


sample_number, k, r = 2**10, 2, 4
data_2d = np.random.randint(0, 500, (sample_number, 2))
artificial_cluster_size = 2**6
artificial_cluster = np.random.randint(100, 130, (artificial_cluster_size, 2))
data_2d = np.vstack((data_2d, artificial_cluster))
sample_number += artificial_cluster_size
start_time = time.time()

result = find(data_2d, (0, 25), 50, 0.1, 0.5, 2**-20, 3)
print result
print "run-time: %.2f seconds" % (time.time() - start_time)

zipped_data = map(list, zip(*data_2d))
plt.scatter(*zipped_data, c='b')
plt.show()

