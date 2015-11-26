import basicdp
import math
import matplotlib.pyplot as plt


def evaluate(data, range_max_value, quality_function, quality_promise, approximation, eps, delta,
             intervals_bounding, max_in_interval):
    # TODO go through variables names and see if they are more or less accurate
    # TODO and maybe change some of the 'k' 'j' 'i'
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
    first_intervals = [range(range_max_value_tag)[i:i + good_interval]
                       for i in range(0, range_max_value_tag, good_interval)]
    second_intervals = [range(good_interval/2, range_max_value_tag)[i:i + good_interval]
                        for i in range(0, range_max_value_tag-good_interval/2, good_interval)]

    # TODO temp - remove later
    # plotting for testing
    fq = [max_in_interval(data, i) for i in first_intervals]
    plt.plot(range(len(fq)), fq, 'bo', range(len(fq)), fq, 'r')
    lower_bound = max(fq) - math.log(1/delta)/eps
    plt.axhspan(lower_bound, lower_bound, color='green', alpha=0.5)
    plt.show()

    # step 9 ( using 'dist' algorithm)
    print "step 9"
    first_chosen_interval = basicdp.a_dist(data, first_intervals, max_in_interval, eps, delta)
    second_chosen_interval = basicdp.a_dist(data, second_intervals, max_in_interval, eps, delta)

    print type(first_chosen_interval)
    print type(second_chosen_interval)
    if type(first_chosen_interval) == str or type(second_chosen_interval) == str:
        raise ValueError("stability problem, try taking more samples!")

    # step 10
    print "step 10"
    return basicdp.exponential_mechanism(data, first_chosen_interval + second_chosen_interval,
                                         extended_quality_function, eps)

