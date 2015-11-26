"""
build in order to stay organized and to afford 'playing' with the algorithms implemented
helper module containing:
 a) random distributed data
 b) quality functions
 c) methods related to differential privacy notions
"""
import numpy as np
from collections import deque


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
    greater_than = sum(e > range_element for e in data)
    less_than = sum(e < range_element for e in data)
    return min(greater_than, less_than)


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
        return min_max_maximum_quality(data,range(max_range + 1))
    ceil_data = [np.ceil(i) for i in data]
    floor_data = [np.floor(i) for i in data]
    rounded_data = floor_data + ceil_data
    points_of_interest = list(set(filter(lambda x: 0 <= x <= max_range, rounded_data)))
    start_point = [min(quality_minmax(data, i), quality_minmax(data, i+2**j-1)) for i in points_of_interest]
    end_point = [min(quality_minmax(data, i-2**j+1), quality_minmax(data, i)) for i in points_of_interest]
    return max(start_point+end_point)


def min_max_maximum_quality(data, domain):
    greater_than = np.count_nonzero(data > domain[0])
    # greater_than = np.size(np.where())
    less_than = len(data) - greater_than
    after_domain = np.count_nonzero(data > domain[len(domain) - 1])
    # after_domain = np.size(np.where(data > domain[len(domain) - 1]))
    while greater_than > less_than and greater_than > after_domain:
        greater_than -= 1
        less_than += 1
    return greater_than


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


def __make_point_data(data_size, specify_spike):
    if specify_spike == -1:
        spike = np.random.randint(data_size, size=1)
    else:
        spike = specify_spike
    if spike > data_size:
        raise ValueError('ERR: spike index is bigger than the data length')
    point_data = [0]*data_size
    point_data[spike] = 1
    return point_data


def __make_threshold_data(data_size, specify_threshold):
    if specify_threshold == -1:
        threshold = np.random.randint(data_size, size=1)[0]
    else:
        threshold = specify_threshold
    if threshold > data_size:
        raise ValueError('ERR: threshold index is bigger than the data length')
    threshold_data = [1]*threshold+[0]*(data_size-threshold)
    return threshold_data


def get_random_data(data_size, distribution_type='normal', pivot=0, specify_parameter=-1):
    """
    get a simple random data set
    :param data_size: number of elements
    :param distribution_type: specify the data type or remain empty to get a normal distributed one
    :return: random data set from a specific tpe
    """

    # TODO is this a good design??
    # lazy switch to get the desirable distribution
    data_switch = {
        'normal': lambda: np.random.normal(pivot, data_size / 6.0, data_size),
        'laplace': lambda: np.random.laplace(pivot, data_size / 100.0, data_size),
        'bimodal': lambda: np.concatenate([np.random.exponential(data_size * 0.07, data_size / 2),
                                   np.random.normal(data_size / 2, data_size / 10.0, data_size / 2)]),
        'uniform': lambda: np.random.uniform(pivot, data_size, data_size),
        'point': lambda: __make_point_data(data_size, specify_parameter),  # POINT_d
        'threshold': lambda: __make_threshold_data(data_size, specify_parameter),  # THRESH_d
    }
    # if user call for unknown data-type return a normal distributed one
    return data_switch.get(distribution_type, data_switch['normal'])()


def get_labeled_sample(data, sample_size):
    """
    get a rando labled sample from a labeled data set
    :param data: list of values
    :param sample_size: the number of samples to return
    :return: list of two lists -
    one containing the indexes sampled
    and the second containing the values in those indexes
    """
    data_size = len(data)
    sampled_data_x = sorted(np.random.choice(data_size, sample_size))
    sampled_data_y = [data[sampled_data_x[i]] for i in xrange(len(sampled_data_x))]
    sampled_data = [sampled_data_x, sampled_data_y]
    return sampled_data


def make_neighbour_set(data, label_type='float'):
    """
    create new data set that differ from the data exactly in one element
    used to check privacy under the classic differential privacy definition
    :param data: the original data set
    :return: data set that differ from the data exactly in one element
    """
    target_index = np.random.randint(len(data))

    # remove random element
    neighbor_data = np.delete(data, target_index)

    # TODO adding necessary?
    # add random element
    random_element = {
        'float': np.random.uniform(min(data), max(data)),
        'int': np.random.randint(min(data), max(data)),
        'binary': np.random.randint(2),
    }
    neighbor_data = np.insert(neighbor_data, target_index, random_element.get(label_type))
    return neighbor_data


# TODO needed?
def databases_distance(data_1, data_2):
    return sum(data_1 != data_2)


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


