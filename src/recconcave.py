import basicdp
import math


def reconcave_basis(T, q, eps, S):
    """recursion basis for the reconcave procedure - execute the exponential mechanism
    note that the parameters r,alpha,delta and N are not being used
    reconcave_basis(solution set size, quality function of sensitivity 1, eps privacy parameter, solution set)
    """
    return basicdp.exponential_mechanism(S, range(T + 1), q, eps)


def evaluate(range_max_value, quality_function, quality_promise, alpha, eps, delta, data, recursion_bound):
    if recursion_bound == 1 or range_max_value <= 32:
        return reconcave_basis(range_max_value, quality_function, quality_promise, alpha, eps, delta, data)
    else:
        recursion_bound -= 1

    log_of_range = math.ceil(math.log(range_max_value, 2))
    range_max_value_tag = 2 ** log_of_range

    def recursive_quality_function(data_base, i):
        if range_max_value < i <= range_max_value_tag:
            min(0, quality_function(data_base, range_max_value))
        else:
            return quality_function(data_base, i)

    # TODO finish this half beaked function
    def intervals_quality(data_base, j):
        intervals_sized_2_to_the_j = [range(range_max_value_tag)[i:i+(2 ** j)] for i in range(range_max_value-(2 ** j)+2)]
        qualified_intervals = [[recursive_quality_function(data,i) for i in inner_list]
                               for inner_list in intervals_sized_2_to_the_j]
        # min_quality_in_intervals = lambda d i : [recursive_quality_function(data_base,i) for i in ]
        # max(map(min,qualified_intervals))
        return

    # def L(s, j):
    #     if 0 <= j <= logT:
    #         return max(min(q_new(s,i)))

    return
