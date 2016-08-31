import numpy as np
from basicdp import choosing_mechanism_big, above_threshold, noisy_avg
from collections import Counter
from jl import johnson_lindenstrauss_transform_init as jl_init
from functools import partial
from scipy.spatial import distance
from numpy.random import laplace
from random import choice
from numpy.linalg import norm


def __box_containing_point__(point, partition, dimension, side_length):
    """
    finds the 'box' containing a given points
    :param point: point in R^dimension as list of coordinates
    :param partition: the boxes partitioning of the space, given by the shift and the size of the 'boxes'
    :param dimension: the dimension of the space which the points are taken from
    :param side_length: the size of the boxes' side
    :return: the axis-aligned box from the given grid, which contains the input point.
     in other words - a list of coordinates such that for each i, it holds that point[i] is in
     the interval  [result[i], result[i] + partition[i])
    """
    try:
        return tuple(np.floor((point[i]-partition[i]) / side_length) for i in xrange(dimension))
    # assuming the exception is due to the data being 1-dimensional
    except IndexError:
        return np.floor((point-partition) / side_length)


def __interval_containing_point__(point, side_length):
    """
    finds the interval containing a given value
    :param point: float. value of points in R^1
    :param side_length: length of intervals
    :return: float. the interval [result, result + side_length) contains the input point
    """
    return np.floor(point / side_length)


def histograms(data, dimension, shift, side, eps, delta):
    """
    Based on Theorem 2.5 from "Locating a Small Cluster Privately"
    by Kobbi Nissim, Uri Stemmer, and Salil Vadhan. PODS 2016.
    find parts of R^d that contain a lot of data-points
    when partitioning R^d into boxes of the same size
    the boxes partitioning is given by the shift and the size of the 'boxes'
    :param data: list of points in R^dimension
    :param dimension: the dimension of the space which the points are taken from
    :param shift: the partition's shift, the i-th value represents the shift in the i-th axis
    :param side: the side-length of each 'box' in the partition
    :param eps: privacy parameter
    :param delta: privacy parameter
    :return: parts of the partition that contain a lot of data-points
    """
    my_box = partial(__box_containing_point__, partition=shift, dimension=dimension, side_length=side)
    # those are the parts of the partition that have at least one point
    #  when each box appears as many times as the numbers of points in it
    boxes = [my_box(point) for point in data]
    boxes_quality = Counter(boxes)
    non_zero = False
    for b in boxes_quality:
        boxes_quality[b] += laplace(0, 2/eps, 1)[0]
        if boxes_quality[b] < 2*np.log(2/delta)/eps:
            boxes_quality[b] = 0
        # the current boxes_quality won't be '0' so the process can return an answer
        elif not non_zero:
            non_zero = True
    if not non_zero:
        raise ValueError('No high quality box')
    return max(boxes_quality)


def find(data, number_of_points, data_dimension, radius, points_in_ball,
         failure, approximation, eps, delta, shrink=False, use_histograms=False):
    # TODO number_of_points is redundant
    """
    Given a data set, desired number of points and a radius finds the center a cluster with approximately
    that number of points and approximately that radius
    :param data: list of points in R^dimension
    :param number_of_points:  number of points in the input data
    :param data_dimension: the dimension of the space which the points are taken from
    :param radius: the radius of cluster to find
    :param points_in_ball: the number of desired points in the resulting cluster
    :param approximation: 0 < float < 1. the approximation level of the result
    :param failure: 0 < float < 1. chances that the procedure will fail to return an answer
    :param eps: float > 0. privacy parameter
    :param delta: 1 > float > 0. privacy parameter
    :param shrink: boolean. default=False. if set to True will try to reduce the dimension to
    obtain a better answer (not relevant in dimension < 600)
    :param use_histograms: boolean. default=False. if set to True will use Theorem 2.5 from the paper
    instead of using the choosing-mechanism (as in the older versions of the paper)
    :return: the center a cluster with approximately that number of points and approximately that radius
    """
    # step 1
    # print "step 1"
    if shrink:
        new_dimension = int(46 * np.log2(2 * number_of_points / failure))
    else:
        new_dimension = data_dimension
    box_side_length = 300 * radius

    # step 2
    # print "step 2"
    if shrink:
        transform = jl_init(data_dimension, new_dimension)
        projected_data = transform(data)
    else:
        def transform(x): return x
        projected_data = data
    threshold = points_in_ball - 100 * np.log2(2 * number_of_points / failure) / eps
    # print "the threshold is: %f" % threshold
    above_thresh = above_threshold(projected_data, threshold, eps/4.0)

    # step 3
    # print "step 3"
    boxes_shift = []  # just to make sure the list is defined in step 7
    found_max = False
    tries = 2 * number_of_points * int(np.log2(1 / failure)) / failure
    # print "maximum no. of tries: %d" % tries
    while not found_max and tries > 0:
        boxes_shift = np.random.uniform(0, box_side_length, new_dimension)

        # step 5
        # print "step 5"

        def partition_quality(data_base):
            # TODO seems like I am ignoring 0-quality elements. Need fix?
            boxes = (__box_containing_point__(p, boxes_shift, new_dimension, box_side_length) for p in data_base)
            c = Counter(boxes)
            return c[max(c, key=c.get)]

        # print "maximum number of points in a single box: %d" % partition_quality(projected_data)
        find_best_box = above_thresh(partition_quality)
        if find_best_box == 'up':
            found_max = True
        else:  # find_best_box == 'bottom'
            tries -= 1

    # step 6
    # print "step 6"
    if not found_max:
        return -1

    # step 7
    # print "step 7"
    box_containing_point_our_case = partial(__box_containing_point__, partition=boxes_shift, dimension=new_dimension,
                            side_length=box_side_length)
    boxes = (box_containing_point_our_case(point) for point in projected_data)
    boxes_quality = Counter(boxes)

    # we add data_base to the signature to match the requirements of choosing_mechanism
    def box_quality(data_base, box):
        return boxes_quality[box]

    boxes_set = list(set(box_containing_point_our_case(p) for p in projected_data))

    if use_histograms:
        best_box = histograms(projected_data, new_dimension, boxes_shift, box_side_length, eps / 4., delta / 4.)
    else:
        best_box = choosing_mechanism_big(projected_data, boxes_set, box_quality, 1, approximation, failure, eps/4.0, delta/4.0)
        if type(best_box) == str:
            raise ValueError("choosing mechanism returned 'bottom'")

    # the first reshape is due to the signature of the transform method
    # the second reshape returns the box to the original structure so we can compare to the best_box
    points_in_best_box = [p for p in data
                          if box_containing_point_our_case(transform(p.reshape(1, data_dimension)).reshape(new_dimension,)) == best_box]

    # print len(points_in_best_box)
    # print "step 8"
    interval_length = 450 * radius * np.sqrt(new_dimension)
    center_box = []
    for axis in xrange(data_dimension):
        # TODO check if there a build-in function for projecting
        projection_on_axis = np.array([d[axis] for d in points_in_best_box])
        eps_tag = eps / np.sqrt(data_dimension * np.log(8/delta)) / 10.0
        delta_tag = delta / data_dimension / 8.0
        if use_histograms:
            best_interval = histograms(projection_on_axis, 1, 0, interval_length, eps_tag, delta_tag)
            extended_interval = ((best_interval - 1) * interval_length, (best_interval + 2) * interval_length)
            center_box.append(extended_interval)
        else:
            axis_projection = np.array([__interval_containing_point__(d[axis], interval_length)
                                        for d in points_in_best_box])
            axis_counter = Counter(axis_projection)

            def interval_quality(data_base, interval_index):
                return axis_counter[interval_index]

            # TODO what is the failure and approximation parameter?
            # TODO should I use the 'sparse' version?
            best_interval = choosing_mechanism_big(projected_data, axis_projection, interval_quality,
                                                   1, approximation, failure, eps_tag, delta_tag)
            try:
                extended_interval = ((best_interval-1) * interval_length, (best_interval+2) * interval_length)
                center_box.append(extended_interval)
            except TypeError:
                raise ValueError("choosing mechanism returned 'bottom'")

    # print "step 9"
    center_of_chosen_box = [(i[1]-i[0])/2. for i in center_box]
    try:
        chosen_ball = [tuple(p) for p in data if distance.euclidean(center_of_chosen_box, p) <= interval_length*3]
    # TODO when doea this error rise?
    except ValueError:
        raise ValueError("something wrong! the center found is %s" % (str(center_of_chosen_box)))

    if not chosen_ball:
        print "chosen ball is empty!"
        return center_of_chosen_box
        # TODO when done - change the return to the error
        # raise ValueError("chosen ball is empty! the center found is %s" % (str(center_of_chosen_box)))

    # step 10
    # print "step 10"
    def predictor(x):
        return x in chosen_ball

    sensitivity = max(norm(v) for v in chosen_ball)
    # best_box, box_quality(data, best_box), center_box, chosen_ball
    return noisy_avg(chosen_ball, predictor, sensitivity, data_dimension, eps/4., delta/4.)

