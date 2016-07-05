from src.good_center import find
from numpy import array
from numpy.random import normal
import numpy as np
import time
from sklearn.metrics.pairwise import euclidean_distances as distances
import sklearn.datasets as dss


sample_number, dimension = 1000, 1000
blobs = dss.make_blobs(sample_number, dimension, cluster_std=5)
blob = blobs[0]
"""
sample_number, dimension = 500, 1000
center = 100
data_2d = array([int(i) for i in normal(center, 5, sample_number*dimension)])
data_2d = data_2d.reshape((sample_number, dimension))
"""
radius, desired_points = 75, 100

start_time = time.time()
result = find(blob, sample_number, dimension, radius, desired_points, 0.1, 0.05, 0.5, 2**-20, shrink=True)

print result

print "run-time: %.2f seconds" % (time.time() - start_time)



