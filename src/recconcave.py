import basicdp
import math


def reconcave_basis(T, q, eps, S):
    """recursion basis for the reconcave procedure - execute the exponential mechanism
    note that the parameters r,alpha,delta and N are not being used
    reconcave_basis(solution set size, quality function of sensitivity 1, eps privacy parameter, solution set)
    """
    return basicdp.exponential_mechanism(S, range(T + 1), q, eps)


def evaluate(range_max_value, quality_function, quality_promise, approximation, eps, delta, data, recursion_bound):
    if recursion_bound == 1 or range_max_value <= 32:
        return reconcave_basis(range_max_value, quality_function, quality_promise, approximation, eps, delta, data)
    else:
        recursion_bound -= 1

    # step 2
    log_of_range = math.ceil(math.log(range_max_value, 2))
    range_max_value_tag = 2 ** log_of_range

    def extended_quality_function(data_base, i):
        if range_max_value < i <= range_max_value_tag:
            min(0, quality_function(data_base, range_max_value))
        else:
            return quality_function(data_base, i)

    # step 3
    def intervals_quality(data_base, j):
        if j == log_of_range + 1:
            return min(0, intervals_quality(data_base, j-1))
        intervals_sized_2_to_the_j = [range(range_max_value_tag)[i:i+(2 ** j)]
                                      for i in range(range_max_value-(2 ** j)+2)]
        qualified_intervals = [[extended_quality_function(data_base, i) for i in inner_list]
                                   for inner_list in intervals_sized_2_to_the_j]
        return max(map(min, qualified_intervals))

    # step 4
    def recursive_quality_function(data_base, j):
        return min(intervals_quality(data_base, j) - (1 - approximation) * quality_promise,
                   quality_promise-intervals_quality(data_base, j + 1))

    # step 5
    recursive_quality_promise = quality_promise * approximation / 2

    # step 6 - recursion call
    recursion_returned = evaluate(log_of_range, recursive_quality_function, recursive_quality_promise, 1/4,
                                  eps, delta, data, recursion_bound)
    return
