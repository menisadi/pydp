import basicdp
import math
from examples import __build_intervals_set__
import matplotlib.pyplot as plt


# TODO check endpoints of interval along the code
def evaluate(data, range_max_value, quality_function, quality_promise, approximation, eps, delta,
             intervals_bounding, max_in_interval, use_exponential=True):
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
    # print "step 2"
    log_of_range = int(math.ceil(math.log(range_max_value, 2)))
    range_max_value_tag = 2 ** log_of_range

    def extended_quality_function(data_base, j):
        if range_max_value < j <= range_max_value_tag:
            return min(0, quality_function(data_base, range_max_value))
        else:
            return quality_function(data_base, j)

    # step 4
    # print "step 4"

    def recursive_quality_function(data_base, j):
        return min(intervals_bounding(data_base, range_max_value_tag, j) - (1 - approximation) * quality_promise,
                   quality_promise-intervals_bounding(data_base, range_max_value_tag, j + 1))

    # step 6
    # print "step 6"
    recursion_returned = basicdp.exponential_mechanism(data, range(log_of_range+1), recursive_quality_function, eps)

    good_interval = 8 * (2 ** recursion_returned)
    # print "good interval: %d" % good_interval

    # step 7
    # print "step 7"
    first_intervals = __build_intervals_set__(data, good_interval, range_max_value_tag)
    second_intervals = __build_intervals_set__(data, good_interval, range_max_value_tag, True)

    # step 9 ( using 'dist' algorithm )
    # print "step 9"
    if use_exponential:
        first_chosen_interval = basicdp.exponential_mechanism(data, first_intervals, max_in_interval, eps)
        second_chosen_interval = basicdp.exponential_mechanism(data, second_intervals, max_in_interval, eps)
        test1 = basicdp.a_dist(data, first_intervals, max_in_interval, eps, delta)
        test2 = basicdp.a_dist(data, second_intervals, max_in_interval, eps, delta)
        print test1,test2
        print first_chosen_interval, second_chosen_interval
        if type(test1) == str or type(test2) == str:
            print "dist would have fail"
            print first_chosen_interval
            qualified_domain = [max_in_interval(data, i) for i in first_intervals]
            ys = [i for i in qualified_domain]
            xs = [i[0] for i in first_intervals]
            h1_score = max(qualified_domain)
            h1 = first_intervals[qualified_domain.index(h1_score)] 
            qualified_domain.remove(h1_score)
            first_intervals.remove(h1)
            h2_score = max(qualified_domain)
            h2 = first_intervals[qualified_domain.index(h2_score)]
            print h1, h2
            plt.plot(xs, ys, 'o')
            plt.show()
            print second_chosen_interval
            qualified_domain = [max_in_interval(data, i) for i in second_intervals]
            ys = [i for i in qualified_domain]
            xs = [i[0] for i in second_intervals]
            h1_score = max(qualified_domain)
            h1 = second_intervals[qualified_domain.index(h1_score)] 
            qualified_domain.remove(h1_score)
            second_intervals.remove(h1)
            h2_score = max(qualified_domain)
            h2 = second_intervals[qualified_domain.index(h2_score)]
            print h1, h2
            plt.plot(xs, ys, 'o')
            plt.show()
    else:
        first_chosen_interval = basicdp.a_dist(data, first_intervals, max_in_interval, eps, delta)
        second_chosen_interval = basicdp.a_dist(data, second_intervals, max_in_interval, eps, delta)

    if type(first_chosen_interval) == str or type(second_chosen_interval) == str:
        raise ValueError("stability problem, try taking more samples!")

    # step 10
    # print "step 10"
    first_chosen_interval_as_list = range(first_chosen_interval[0], first_chosen_interval[1]+1)
    second_chosen_interval_as_list = range(second_chosen_interval[0], second_chosen_interval[1]+1)
    return basicdp.exponential_mechanism(data, first_chosen_interval_as_list + second_chosen_interval_as_list,
                                         extended_quality_function, eps)

