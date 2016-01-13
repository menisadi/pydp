import numpy as np
from collections import deque


def iterlen(it):
    """
    a length of generator
    :param it: generator object
    :return: the number of items in it
    """
    return sum(1 for _ in it)


def quality_median(data, range_element):
    """
    sensitivity-1 quality function
    used to find the median fo the data
    quality_median( data , range_element )
    :return: the "distance" of range_element from the median of the data
    """
    greater_than = sum(e >= range_element for e in data)
    less_than = sum(e < range_element for e in data)
    return -max(0, len(data) / 2 - min(greater_than, less_than))


def bulk_quality_median(data, domain):
    """
    sensitivity-1 bulk quality function
    used to find the median fo the data
    quality_median( data , domain )
    :return: list of "distances" of every domain elements from the median of the data
    """
    greater_than = len(data)
    less_than = 0
    domain_que = deque(sorted(data))
    qualities = []
    data_next = min(domain)-1
    while len(domain_que) > 0:
        data_prev = data_next
        data_next = domain_que.popleft()
        qualities.append([-max(0, len(data) / 2 - min(greater_than, less_than))
                          for i in domain if data_prev < i <= data_next])
        greater_than -= 1
        less_than += 1
    qualities.append([-max(0, len(data) / 2 - min(greater_than, less_than))
                      for i in domain if data_next < i])
    # qualities is a list of lists of qualities so:
    # return flatted qualities list
    return [item for sub_list in qualities for item in sub_list]


# for rec_concave testing
# TODO not in use!
# TODO can we use quality_median instead? can we delete this?
def quality_minmax(data, range_element):
    """
    sensitivity-1 quality function
    used to find the minmax or median for the data
    quality_minmax( data , range_element )
    :return: the minimum between the amount of data above the element and the data below
    """
    greater_than = iterlen(x for x in data if x > range_element)
    less_than = iterlen(x for x in data if x < range_element)
    return min(less_than, greater_than)


def bulk_quality_minmax(data, domain):
    """
    sensitivity-1 bulk quality function
    used to find a minmax or median for the data
    bulk_quality_minmax( data , domain )
    :return: the minimum between the amount of data above the element and the data below
    """
    greater_than = len(data)
    less_than = 0
    domain_que = deque(sorted(data))
    qualities = []
    data_next = min(domain) - 1
    while len(domain_que) > 0:
        data_prev = data_next
        data_next = domain_que.popleft()
        qualities.append([min(greater_than, less_than)
                          for i in domain if data_prev < i <= data_next])
        greater_than -= 1
        less_than += 1
    qualities.append([min(greater_than, less_than)
                      for i in domain if data_next < i])
    # qualities is a list of lists of qualities so:
    # return flatted qualities list
    return [item for sub_list in qualities for item in sub_list]


def min_max_intervals_bounding(data, max_range, j):
    if j == 0:
        return min_max_maximum_quality(data, (0, max_range))

    ceil_data = set(np.ceil(x) for x in data)
    floor_data = set(np.floor(x) for x in data)
    rounded_data = floor_data.union(ceil_data)
    points_of_interest = [x for x in rounded_data if x >= 2**j - 1 or x <= max_range - 2**j + 1]
    before = set(y - 2**j + 1 for y in points_of_interest if y >= 2**j - 1)
    after = set(y + 2**j - 1 for y in points_of_interest if y <= max_range - 2**j + 1)

    points_to_qualify = sorted(list(set(points_of_interest).union(before, after)))
    if len(points_to_qualify) == 0:
        return 0
    else:
        interest_qualities = bulk_quality_minmax(data, points_to_qualify)

        def __quality__(d):
            ind = points_to_qualify.index(d)
            return interest_qualities[ind]

        start_point = [min(__quality__(x), __quality__(x+2**j-1))
                       for x in points_of_interest if x <= max_range - 2**j + 1]
        end_point = [min(__quality__(x-2**j+1), __quality__(x))
                     for x in points_of_interest if x >= 2**j - 1]
        return max(start_point+end_point)


def min_max_maximum_quality(data, interval):
    greater_than = iterlen(x for x in data if x > interval[0])
    less_than = len(data) - greater_than
    after_domain = iterlen(x for x in data if x > interval[1])
    while greater_than > less_than and greater_than > after_domain:
        greater_than -= 1
        less_than += 1
    return min(less_than, greater_than)


# TODO not in use!
def quality_mode(data, range_element):
    return sum([d == range_element for d in data])


def quality_point_mode(data, range_element):
    return sum([(data[0][i], data[1][i]) == (range_element, 1) for i in xrange(len(data[0]))])


# page 9 -  used for a proper private learner for POINT_d
def point_concept_quality(data, point):
    """
    sensitivity-1 quality function
    used to find a point concept labeling the data
    :param data: labeled sample from a POINT_d data set.
                represented as list of indexes and list of their values in the original data set
    :param point: point concept index
    :return: the number of times (point, 1) appears in the data
    """
    indexes = [i for i, e in enumerate(data[0]) if e == point]
    return sum([data[1][i] for i in indexes])


# TODO maybe think about this direction of generalization
def concept_quality(sampled_data, concept):
    return sum([sampled_data[1][i] == concept(sampled_data[0][i]) for i in xrange(len(sampled_data[0]))])


def concept_query(data, concept):
    return sum([1 for i in data if concept(i) == 1])/float(len(data))


# TODO choose approach
# first approach
def interval_threshold_quality(sampled_data, threshold_index):
    # assuming that sampled_data is two list of the same length - one of x's and one of y's
    xs = sampled_data[0]
    ys = sampled_data[1]
    # sum the number of indexes which 'agree' to the give threshold
    return sum([(ys[i] == 0 and xs[i] >= threshold_index) or
                (ys[i] == 1 and xs[i] < threshold_index) for i in xrange(len(xs))])


# second approach
def threshold_function(index, threshold):
    if index < threshold:
        return 1
    else:
        return 0


# second approach
def interval_threshold_quality2(sampled_data, threshold_index):
    def concept(x):
        return threshold_function(x, threshold_index)
    return concept_quality(sampled_data, concept)

