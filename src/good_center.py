import numpy as np
import basicdp
from collections import Counter
from jl import johnson_lindenstrauss_transform as jl


def __box_point(point, partition, k, side_length):
    return tuple(np.floor((point[i]-partition[i]) / side_length) for i in xrange(k))


def find(data, number_of_points, dimension, radius, points_in_ball, failure, eps, delta, shrink=False):
    # step 1
    if shrink:
        k = int(46 * np.log(2 * number_of_points / failure))
    else:
        k = dimension
    box_side_length = 6 * k * radius

    # step 2
    if shrink:
        projected_data = jl(data, dimension, k)
    else:
        projected_data = data
    threshold = points_in_ball - 100 * np.log(2 * number_of_points / failure) / eps
    above_thresh = basicdp.above_threshold(projected_data, threshold, eps/4.0)

    # step 3
    found_max = False
    tries = int(np.log(1 / failure))
    while not found_max and tries > 0:
        boxes_shift = np.random.uniform(0, box_side_length, k)

        # step 5
        def partition_quality(data_base):
            boxes = (__box_point(p, boxes_shift, k, box_side_length) for p in data_base)
            c = Counter(boxes)
            return c[max(c, key=c.get)]

        best_box = above_thresh(partition_quality)
        if type(best_box) != str:
            found_max = True
        else:
            tries -= 1

    # step 6
    if not found_max:
        return -1

    # step 7
    def box_quality(data_base, box):
        boxes = (__box_point(p, boxes_shift, k, box_side_length) for p in data_base)
        c = Counter(boxes)
        return c[box]

    boxes_set = set(__box_point(p, boxes_shift, k, box_side_length) for p in data)
    best_box = basicdp.choosing_mechanism(projected_data,boxes,box_quality,1,0.2,failure,eps/3.0,delta/3.0)
    return

