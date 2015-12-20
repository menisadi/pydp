import basicdp
import math
from collections import deque


def __build_intervals_set__(data_base, interval_length, range_max, shift = False):
    # assuming all the data is non-negative
    data_que = deque(sorted(data_base))
    list_of_intervals = []
    data_next = -1
    while len(data_que) > 0 and data_next <= range_max:
        data_next = data_que.popleft()
        next_relevant_interval_start = (int(data_next - shift * interval_length/2) / interval_length) \
                                       * interval_length + shift*interval_length/2
        next_relevant_interval = (next_relevant_interval_start, next_relevant_interval_start + interval_length)
        list_of_intervals.append(next_relevant_interval)
        while data_next < next_relevant_interval[1] and len(data_que) > 0:
            data_next = data_que.popleft()

    return list_of_intervals


# TODO check endpoints of interval along the code
def evaluate(data, range_max_value, quality_function, quality_promise, approximation, eps, delta,
             intervals_bounding, max_in_interval):
    """
    RecConcave algorithm for the specific case of N=2
    :param data:
    :param range_max_value: maximum possible output (the minimum output is 0)
    :param quality_function:
    :param quality_promise:
    :param approximation: approximation parameter (from 0 to 1)
    :param eps, delta: privacy parameters
    :param intervals_bounding: function L(data,domain_element)
    :param max_in_interval: function u(data,interval) that returns the maximum of quality_function(data,j)
    for j in the interval
    :return:
    """

    # step 2
    print "step 2"
    log_of_range = int(math.ceil(math.log(range_max_value, 2)))
    range_max_value_tag = 2 ** log_of_range

    def extended_quality_function(data_base, j):
        if range_max_value < j <= range_max_value_tag:
            return min(0, quality_function(data_base, range_max_value))
        else:
            return quality_function(data_base, j)

    # step 4
    print "step 4"

    def recursive_quality_function(data_base, j):
        return min(intervals_bounding(data_base, range_max_value_tag, j) - (1 - approximation) * quality_promise,
                   quality_promise-intervals_bounding(data_base, range_max_value_tag, j + 1))

    # step 6
    print "step 6"
    recursion_returned = basicdp.exponential_mechanism(data, range(log_of_range+1), recursive_quality_function, eps)

    good_interval = 8 * (2 ** recursion_returned)
    print "good interval: %d" % good_interval

    # step 7
    print "step 7"
    first_intervals = __build_intervals_set__(data, good_interval, range_max_value_tag)
    # old_first_intervals =  [(i, i+good_interval) for i in xrange(0, range_max_value_tag, good_interval)]
    second_intervals = __build_intervals_set__(data, good_interval, range_max_value_tag, True)
    # old_second_intervals = [(i, i+good_interval) for i in xrange(good_interval/2, range_max_value_tag, good_interval)]

    # step 9 ( using 'dist' algorithm )
    print "step 9"
    first_chosen_interval = basicdp.a_dist(data, first_intervals, max_in_interval, eps, delta)
    # old_first_chosen_interval = basicdp.a_dist(data, old_first_intervals, max_in_interval, eps, delta)
    second_chosen_interval = basicdp.a_dist(data, second_intervals, max_in_interval, eps, delta)
    # old_second_chosen_interval = basicdp.a_dist(data, old_second_intervals, max_in_interval, eps, delta)
    # print first_chosen_interval, old_first_chosen_interval
    # print second_chosen_interval, old_second_chosen_interval

    # print type(first_chosen_interval)
    # print type(second_chosen_interval)
    if type(first_chosen_interval) == str or type(second_chosen_interval) == str:
        raise ValueError("stability problem, try taking more samples!")

    # step 10
    print "step 10"
    first_chosen_interval_as_list = range(first_chosen_interval[0], first_chosen_interval[1]+1)
    second_chosen_interval_as_list = range(second_chosen_interval[0], second_chosen_interval[1]+1)
    return basicdp.exponential_mechanism(data, first_chosen_interval_as_list + second_chosen_interval_as_list,
                                         extended_quality_function, eps)

