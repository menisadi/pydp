import src.good_radius as gr
import src.good_center as gc
from numpy.random import randint, normal
import numpy as np
import matplotlib.pyplot as plt
import time
from numpy import array
from scipy.spatial.distance import euclidean


sample_number, k, r = 2**12, 2, 4
center = 100
data_2d = array([int(i) for i in normal(center, 50, sample_number)]).reshape((sample_number/2, 2))

start_time = time.time()
domain, desired_amount_of_points = (0, 100), 2000
approximation, failure, eps, delta, promise = 0.1, 0.1, 0.5, 2 ** -20, 100
radius = gr.find(data_2d, domain, desired_amount_of_points, failure, eps, delta, promise)
print "the radius: %d" % radius
middle_time = time.time()
print "good-radius run-time: %.2f seconds" % (middle_time - start_time)
center = gc.find(data_2d, sample_number, 2, radius, desired_amount_of_points, failure, approximation, eps, delta)
print "the center: %s" % str(center)
print "good-center run-time: %.2f seconds" % (time.time() - middle_time)
ball = [p for p in data_2d if euclidean(p, center) <= radius]
print "number of points in the resulting ball: %d" % len(ball)

zipped_data = zip(*data_2d)
plt.scatter(*zipped_data, c='b')
zipped_ball = zip(*ball)
plt.scatter(*zipped_ball, c='r')
plt.show()



