"""
naive implementation of rec_concave
generic but not efficient
"""
import basicdp
import math
import matplotlib.pyplot as plt


def __rec_concave_basis__(range_max_value, quality_function, eps, data, bulk=False):
    """recursion basis for the reconcave procedure - execute the exponential mechanism
    note that the parameters r,alpha,delta and N are not being used
    reconcave_basis(solution set size, quality function of sensitivity 1, eps privacy parameter, solution set)
    """
    return basicdp.exponential_mechanism(data, range(int(range_max_value)+1), quality_function, eps, bulk)


# A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
def evaluate(data, range_max_value, quality_function, quality_promise,
             approximation, eps, delta, recursion_bound, bulk=False):
    # TODO fix so it will work
    # TODO add docstring
    # TODO go through variables names
    if recursion_bound == 1 or range_max_value <= 32:
        return __rec_concave_basis__(range_max_value, quality_function, eps, data, bulk)
    else:
        recursion_bound -= 1

    # step 2
    print "step 2"
    log_of_range = int(math.ceil(math.log(range_max_value, 2)))
    range_max_value_tag = 2 ** log_of_range

    if bulk:
        qualities = quality_function(data, range(int(range_max_value)+1))
    else:
        qualities = [quality_function(data, i) for i in range(int(range_max_value)+1)]
    qualities.extend([min(0, qualities[range_max_value]) for _ in xrange(range_max_value, range_max_value_tag)])
    
    def extended_quality_function(j):
        return qualities[j]

    # same but with signature that fits exponential mechanism requirements (used in step 10)
    def extended_quality_function_for_exponential_mechanism(data_set, j):
        return qualities[j]

    # step 3
    print "step 3"

    def intervals_bounding(j):
        if j == log_of_range+1:
            return min(0, intervals_bounding(log_of_range))
        return max(min(extended_quality_function(e) for e in xrange(a, a+2**j-1))
                   for a in xrange(0, range_max_value_tag-2**j+1))

    # step 4
    print "step 4"

    def recursive_quality_function(data_base, range_element):
        return min(intervals_bounding(range_element) - (1 - approximation) * quality_promise,
                   quality_promise - intervals_bounding(range_element + 1))

    # step 5
    print "step 5"
    recursive_quality_promise = quality_promise * approximation / 2
        
    # step 6 - recursion call
    print "step 6 - recursive call"
    recursion_returned = evaluate(data, log_of_range, recursive_quality_function, recursive_quality_promise, 1/4,
                                  eps, delta, recursion_bound, True)
        
    good_interval = 8 * (2 ** recursion_returned)
    print "good interval: %d" % good_interval

    # step 7
    print "step 7"
    first_intervals = [range(range_max_value_tag)[i:i + good_interval]
                       for i in xrange(0, range_max_value_tag, good_interval)]
    
    second_intervals = [range(good_interval/2, range_max_value_tag)[i:i + good_interval]
                        for i in xrange(0, range_max_value_tag-good_interval/2, good_interval)]

    # step 8
    print "step 8"

    def interval_quality(data_base, interval):
        return max([extended_quality_function(j) for j in interval])
    
    # TODO temp - remove later
    # plotting for testing
    fq = [interval_quality(data, i) for i in first_intervals]
    plt.plot(range(len(fq)), fq, 'bo', range(len(fq)), fq, 'r')
    lower_bound = max(fq) - math.log(1/delta)/eps
    plt.axhspan(lower_bound, lower_bound, color='green', alpha=0.5)
    plt.show()

    # step 9 ( using 'dist' algorithm)
    print "step 9"
    first_chosen_interval = basicdp.a_dist(data, first_intervals, interval_quality, eps, delta)
    second_chosen_interval = basicdp.a_dist(data, second_intervals, interval_quality, eps, delta)

    print "first A_dist returned: %s" % str(type(first_chosen_interval))
    print "second A_dist returned: %s" % str(type(second_chosen_interval))

    if type(first_chosen_interval) != list or type(second_chosen_interval) != list:
        raise ValueError('stability problem')

    # step 10
    print "step 10"
    return basicdp.exponential_mechanism(data, first_chosen_interval + second_chosen_interval,
                                         extended_quality_function_for_exponential_mechanism, eps, False)