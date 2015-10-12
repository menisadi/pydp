import basicdp
import math


def rec_concave_basis(range_max_value, quality_function, eps, data):
    """recursion basis for the reconcave procedure - execute the exponential mechanism
    note that the parameters r,alpha,delta and N are not being used
    reconcave_basis(solution set size, quality function of sensitivity 1, eps privacy parameter, solution set)
    """
    return basicdp.exponential_mechanism(data, range(int(range_max_value + 1)), quality_function, eps)


def evaluate(range_max_value, quality_function, quality_promise, approximation, eps, delta, data, recursion_bound):
    # TODO go through variables names and see if they are more or less accurate
    # TODO and maybe change some of the 'k' 'j' 'i'
    if recursion_bound == 1 or range_max_value <= 32:
        return rec_concave_basis(range_max_value, quality_function, eps, data)
    else:
        recursion_bound -= 1

    # step 2
    log_of_range = int(math.ceil(math.log(range_max_value, 2)))
    range_max_value_tag = 2 ** log_of_range

    def extended_quality_function(data_base, j):
        if range_max_value < j <= range_max_value_tag:
            min(0, quality_function(data_base, range_max_value))
        else:
            return quality_function(data_base, j)

    # step 3
    def intervals_bounding(data_base, j):
        # TODO why the +1?
        if j == log_of_range + 1:
            return min(0, intervals_bounding(data_base, j-1))
        # TODO should we add 1 to range input? (here and in step 7)
        intervals_sized_2_to_the_j = [range(range_max_value_tag)[k:k+(2 ** j)]
                                      for k in range(range_max_value-(2 ** j)+2)]
        qualified_intervals = [[extended_quality_function(data_base, k) for k in inner_list]
                               for inner_list in intervals_sized_2_to_the_j]
        return max([min(interval) for interval in qualified_intervals if interval])

    # step 4
    def recursive_quality_function(data_base, j):
        return min(intervals_bounding(data_base, j) - (1 - approximation) * quality_promise,
                   quality_promise-intervals_bounding(data_base, j + 1))

    # step 5
    recursive_quality_promise = quality_promise * approximation / 2

    # step 6 - recursion call
    recursion_returned = evaluate(log_of_range, recursive_quality_function, recursive_quality_promise, 1/4,
                                  eps, delta, data, recursion_bound)
    good_interval = 8 * (2 ** recursion_returned)

    # step 7
    first_intervals = [range(range_max_value_tag)[i:i + good_interval]
                       for i in range(0, range_max_value_tag, good_interval)]
    second_intervals = [range(good_interval/2, range_max_value_tag)[i:i + good_interval]
                        for i in range(0, range_max_value_tag-good_interval/2, good_interval)]

    # step 8
    def interval_quality(data_base, interval):
        return max([extended_quality_function(data_base, j) for j in interval])

    # step 9 ( using 'dist' algorithm)
    first_chosen_interval = basicdp.a_dist(eps, delta, first_intervals, data, interval_quality)
    second_chosen_interval = basicdp.a_dist(eps, delta, second_intervals, data, interval_quality)

    # step 10
    return basicdp.exponential_mechanism(data, first_chosen_interval.append(second_chosen_interval),
                                         extended_quality_function, eps)
