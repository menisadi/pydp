from math import sqrt, log, exp, factorial
from operator import mul
from itertools import compress


def nCr(n, r):
    """
    combinatorial choose function
    :param n: from n
    :param r: choose r
    :return: the number of possible options to choose r objects out of n
    """
    return factorial(n) / factorial(r) / factorial(n-r)


def advanced(eps, delta, delta_tag, k):
    """
    compute the combined privacy parameters of k mechanisms by "advanced composition"
    Dwork, Rothblum, Vadhan - 2010
    :param eps: privacy parameter of each mechanism
    :param delta: privacy parameter of each mechanism
    :param delta_tag: additive lose in the delta parameter
    :param k: number of mechanisms
    :return: privacy parameters for the combined mechanism
    """
    e_tag = sqrt(2*k*log(1/delta_tag))*eps + k*eps*(exp(eps)-1)
    return e_tag, k*delta+delta_tag


def optimal_homogeneous(eps, delta, k):
    """
    compute the optimal combined privacy parameters of k mechanisms
    Kairouz, Oh, Viswanath - 2015
    :param eps: privacy parameter of each mechanism
    :param delta: privacy parameter of each mechanism
    :param k: number of mechanisms
    :return: privacy parameters for the composition of mechanism
    """
    for i in xrange(k/2 + 1):
        sum_i = sum(nCr(k, l)*(exp((k-l)*eps)-exp((k-2*i+l)*eps)) for l in xrange(i))
        delta_i = sum_i/(1-exp(eps))**k
        yield (k-2*i)*eps, 1-(1-delta_i)*(1-delta)**k


# eps is homogeneous
# delta_0 > 0, for i>0 : delta_i = 0
# my calculations, no formal proof included
# TODO fill the blanks
def __optimal_homogeneous_eps__(eps, delta, k):
    """
    compute the optimal combined privacy parameters of k mechanisms
    in the case that the epsilon parameter is homogeneous and all but the first mechanism have pure privacy
    :param eps: privacy parameter of each mechanism
    :param delta: privacy parameter of each mechanism
    :param k:  number of mechanisms
    :return: lazy list of possible privacy parameters for k mechanisms each with (eps,delta)-privacy
    """
    for i in xrange(k/2 + 1):
        sum_i = sum(nCr(k, l)*(exp((k-l)*eps)-exp((k-2*i+l)*eps)) for l in xrange(i))
        delta_i = sum_i/(1-exp(eps))**k
        yield (k-2*i)*eps, 1-(1-delta_i)*(1-delta)


def delta_bound(delta_list):
    """
    lower bound on the combined privacy parameter delta
    :param delta_list: list of delta parameter of each mechanism
    :return: lower bound on the combined privacy parameter delta
    regardless of the eps parameters
    """
    return 1-reduce(mul, [1-d for d in delta_list])


def __splitter__(k):
    """
    compute all possible ways to partition [1...k] into two groups
    :param k: maximum of the partitioned group
    :return: lazy list of all possible ways to partition [1...k] into two groups
    """
    l = range(1, k+1)
    d = k
    for i in range(2**k):
        b = ('{0:0%db}' % d).format(i)
        bb = [i == '1' for i in b]
        split = set(compress(l, bb))
        yield split, set(l)-split


def __sum_part_of_eps_list__(eps_list, part_of_k):
    """
    sum epsilons from a given part of the list [1...k]
    :param eps_list:
    :param part_of_k: part of the list [1...k]
    :return: sum of epsilons from the list if they are included in part_of_k
    """
    return sum(eps_list[i-1] for i in part_of_k)


def optimal_heterogeneous(eps_list, delta_list, eps_g, delta_g):
    """
    compute the optimal combined privacy parameters of k mechanisms
    each with different privacy parameters
    Murtagh, Vadhan - 2015
    :param es, delta_list: lists of privacy parameters
    :param eps_g: desired combined eps privacy parameter
    :param delta_g: desired combined eps privacy parameter
    :return: check if (eg,dg) are indeed legitimate privacy parameters of the composition
    """
    k = len(eps_list)
    sum_eg = sum([max(0,
                      exp(__sum_part_of_eps_list__(eps_list, split[0]))
                      - exp(eps_g)*exp(__sum_part_of_eps_list__(eps_list, split[1])))
                  for split in __splitter__(k)])
    eps_side = 1 / reduce(mul, [1 + exp(e) for e in eps_list])
    delta_side = 1 - (1-delta_g)/(reduce(mul, [1-d for d in delta_list]))
    return eps_side*sum_eg <= delta_side


def find_eg(eps_list, delta_list, fixed_new_delta, t):
    """
    use binary search to find eps_g as in the optimal_heterogeneous method
    :param es, delta_list: lists of privacy parameters
    :param fixed_new_delta: desired combined eps privacy parameter
    :param t: binary search steps limit
    :return: minimal eps_g such (eps_eg, delta_eg) are indeed legitimate privacy parameters of the composition
    """
    k = len(eps_list), len(eps_list)
    i, eg, r_g = sum(eps_list), sum(eps_list), sum(eps_list)
    while t > 0:
        i /= 2.0
        # print t, i , eg
        a = (-1)**optimal_heterogeneous(eps_list, delta_list, eg, fixed_new_delta)
        if a == -1:
            r_g = eg
        eg += a*i
        if eg > k:
            raise ValueError("delta_g too small")
        t -= 1
    return r_g
