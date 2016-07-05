from src.good_center import find
import time
from sklearn.metrics.pairwise import euclidean_distances as distances
import sklearn.datasets as dss
from scipy.spatial.distance import euclidean


sample_number, dimension = 10000, 2
blobs = dss.make_blobs(sample_number, dimension, cluster_std=50)
blob = blobs[0]
radius, desired_points = 50, 2000

start_time = time.time()
result = find(blob, sample_number, dimension, radius, desired_points, 0.1, 0.05, 0.5, 2**-20)

print result
print len([p for p in blob if euclidean(p, result) <= radius])

print "run-time: %.2f seconds" % (time.time() - start_time)



