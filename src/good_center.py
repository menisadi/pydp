import numpy as np
import basicdp
from collections import Counter
from jl import johnson_lindenstrauss_transform_init as jl_init
from functools import partial


def box_point(point, partition, dimension, side_length):
    return tuple(np.floor((point[i]-partition[i]) / side_length) for i in xrange(dimension))


# do we really need jl? it works only if approximately data_dimension > 900
def find(data, number_of_points, data_dimension, radius, points_in_ball, failure, approximation, eps, delta, shrink=False):
    # step 1
    if shrink:
        new_dimension = int(46 * np.log(2 * number_of_points / failure))
    else:
        new_dimension = data_dimension
    box_side_length = 6 * new_dimension * radius

    # step 2
    if shrink:
        transform = jl_init(data_dimension, new_dimension)
        projected_data = transform(data)
    else:
        def transform(x): return x
        projected_data = data
    threshold = points_in_ball - 100 * np.log(2 * number_of_points / failure) / eps
    above_thresh = basicdp.above_threshold(projected_data, threshold, eps/4.0)

    # step 3
    boxes_shift = []  # just to make sure the list is defined in step 7
    found_max = False
    tries = int(np.log(1 / failure))
    while not found_max and tries > 0:
        boxes_shift = np.random.uniform(0, box_side_length, new_dimension)

        # step 5
        def partition_quality(data_base):
            boxes = (box_point(p, boxes_shift, new_dimension, box_side_length) for p in data_base)
            c = Counter(boxes)
            return c[max(c, key=c.get)]

        find_best_box = above_thresh(partition_quality)
        if find_best_box == 'up':
            found_max = True
        else:  # find_best_box == 'bottom'
            tries -= 1

    # step 6
    if not found_max:
        return -1

    # step 7
    def box_quality(data_base, box):
        boxes = (box_point(p, boxes_shift, new_dimension, box_side_length) for p in data_base)
        c = Counter(boxes)
        return c[box]

    our_box_point = partial(box_point, partition=boxes_shift, dimension=new_dimension, side_length=box_side_length)
    boxes_set = list(set(our_box_point(p) for p in data))
    # what is the growth bound?
    best_box = basicdp.choosing_mechanism(projected_data, boxes_set, box_quality, 1,
                                          approximation, failure, eps/3.0, delta/3.0)
    points_in_best_box = [p for p in data
                          if our_box_point(transform(p.reshape(1, data_dimension))) == best_box]

    return best_box, points_in_best_box


n, k, r = 4000, 2, 1
dr2 = np.random.normal(0, 300, (n, 2))
gcn = 10
gc = np.random.normal(6, 0.5, (gcn, 2))
dr2 = np.vstack((dr2, gc))
n += gcn
print find(dr2, n, 2, 1, 5, 0.01, 0.5, 0.5, 0.001)
