import basicdp
import math
import matplotlib.pyplot as plt


def rec_concave_basis(range_max_value, quality_function, eps, data, bulk=False):
    """recursion basis for the reconcave procedure - execute the exponential mechanism
    note that the parameters r,alpha,delta and N are not being used
    reconcave_basis(solution set size, quality function of sensitivity 1, eps privacy parameter, solution set)
    """
    return basicdp.exponential_mechanism(data, range(int(range_max_value)+1), quality_function, eps, bulk)


# A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
def evaluate(data, range_max_value, quality_function, quality_promise, approximation, eps, delta, recursion_bound,
             bulk_interval=False, bulk_interval_quality=None, bulk_quality=False, bulk_quality_function=None):
    # TODO add docstring
    # TODO go through variables names
    if recursion_bound == 1 or range_max_value <= 32:
        if bulk_quality:
            return rec_concave_basis(range_max_value, bulk_quality_function, eps, data, bulk=bulk_quality)
        else:
            return rec_concave_basis(range_max_value, quality_function, eps, data)
    else:
        recursion_bound -= 1

    # step 2
    print "step 2"
    log_of_range = int(math.ceil(math.log(range_max_value, 2)))
    range_max_value_tag = 2 ** log_of_range

    def extended_quality_function(data_base, j):
        if range_max_value < j <= range_max_value_tag:
            return min(0, quality_function(data_base, range_max_value))
        else:
            return quality_function(data_base, j)

    if bulk_quality:
        # TODO improve
        def bulk_extended_quality_function(data_base, domain):
            domain_up_to_range_max = [d for d in domain if d <= range_max_value]
            domain_after_range_max = [d for d in domain if d > range_max_value]
            qualities = bulk_quality_function(data_base, domain_up_to_range_max)  # range(int(range_max_value)+1)
            extra_domain = len(domain_after_range_max)
            qualities.extend([min(0, qualities[len(domain_up_to_range_max)-1])]*extra_domain)
            return qualities

    # step 3
    print "step 3"
    if bulk_interval:
        intervals_bounding = bulk_interval_quality
    else:
        def intervals_bounding(data_base, j):
            # TODO why the +1?
            if j == log_of_range + 1:
                return min(0, intervals_bounding(data_base, j-1))
            intervals_sized_2_to_the_j = [range(range_max_value_tag+1)[k:k+(2 ** j)]
                                          for k in xrange(range_max_value_tag-(2 ** j)+2)]
            qualified_intervals = [[extended_quality_function(data_base, k) for k in inner_list]
                                   for inner_list in intervals_sized_2_to_the_j]
            min_of_intervals = [min(interval) for interval in qualified_intervals if interval]
            return max(min_of_intervals)
    
    # step 4
    print "step 4"

    """
    def recursive_quality_function(data_base, domain):
        if type(domain) == int:
            domain_len = domain
        else:
            domain_len = max(domain)
        return [min(intervals_bounding(data_base, j) - (1 - approximation) * quality_promise,
                    quality_promise-intervals_bounding(data_base, j + 1)) for j in xrange(domain_len+1)]
    """
    def recursive_quality_function(data_base, j):
        return min(intervals_bounding(data_base, j) - (1 - approximation) * quality_promise,
                   quality_promise-intervals_bounding(data_base, j + 1))

    # step 5
    print "step 5"
    recursive_quality_promise = quality_promise * approximation / 2

    # step 6 - recursion call
    print "step 6 - recursive call"
    recursion_returned = evaluate(data, log_of_range, recursive_quality_function, recursive_quality_promise, 1/4,
                                  eps, delta, recursion_bound, False, None, bulk_quality, bulk_quality_function)

    good_interval = 8 * (2 ** recursion_returned)
    print "good interval: %d" % good_interval

    # step 7
    print "step 7"
    first_intervals = [range(range_max_value_tag)[i:i + good_interval]
                       for i in range(0, range_max_value_tag, good_interval)]
    
    second_intervals = [range(good_interval/2, range_max_value_tag)[i:i + good_interval]
                        for i in range(0, range_max_value_tag-good_interval/2, good_interval)]

    # step 8
    print "step 8"

    def interval_quality(data_base, interval):
        if bulk_quality:
            return max(bulk_extended_quality_function(data_base, interval))
        else:
            return max([extended_quality_function(data_base, j) for j in interval])
    
    # TODO temp - remove later
    # plotting for testing
    """
    fq = [interval_quality(data, i) for i in first_intervals]
    plt.plot(range(len(fq)), fq, 'bo', range(len(fq)), fq, 'r')
    lower_bound = max(fq) - math.log(1/delta)/eps
    plt.axhspan(lower_bound, lower_bound, color='green', alpha=0.5)
    plt.show()
    """

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
    """
    if bulk_quality:
        return basicdp.exponential_mechanism(data, first_chosen_interval + second_chosen_interval,
                                             bulk_extended_quality_function, eps, bulk=True)
    else:
    """
    return basicdp.exponential_mechanism(data, first_chosen_interval + second_chosen_interval,
                                             extended_quality_function, eps, bulk=False)
