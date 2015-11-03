import numpy as np


def noisy_max(data, domain, quality_function, eps, bulk=False):
    """Noisy-Max Mechanism
    noisy_max ( data , domain, quality function , privacy parameter )
    :param data:
    :param domain: list of possible results
    :param quality_function:
    :param eps: privacy parameter
    :return: an element of domain with approximately maximum value of quality function
    """

    # compute q(X,i) for all the elements in D
    if bulk:
        qualified_domain = quality_function(data, domain)
    else:
        qualified_domain = [quality_function(data, i) for i in domain]
    # add Lap(1/eps) noise for each element in qualified_domain
    noisy = [q + np.random.laplace(0, 1 / eps, 1) for q in qualified_domain]
    # return element with maximum noisy q(X,i)
    return domain[noisy.index(max(noisy))]


def exponential_mechanism(data, domain, quality_function, eps, bulk=False):
    """Exponential Mechanism
    exponential_mechanism ( data , domain , quality function , privacy parameter )
    :param data:
    :param domain: list of possible results
    :param quality_function:
    :param eps: privacy parameter
    :param bulk:
    :return: an element of domain with approximately maximum value of quality function
    """

    # calculate a list of probabilities for each element in the domain D
    # probability of element d in domain proportional to exp(eps*quality(data,d)/2)
    if bulk:
        qualified_domain = quality_function(data, domain)
        domain_pdf = [np.exp(eps * q / 2) for q in qualified_domain]
    else:
        domain_pdf = [np.exp(eps * quality_function(data, d) / 2) for d in domain]
    normalizer = float(sum(domain_pdf))
    domain_pdf = [d / normalizer for d in domain_pdf]
    normalizer = sum(domain_pdf)
    # for debugging and other reasons: check that domain_cdf indeed defines a distribution
    # use the uniform distribution (from 0 to 1) to pick an elements by the CDF
    if abs(normalizer - 1) > 0.001:
        raise ValueError('ERR: exponential_mechanism, sum(domain_pdf) != 1.')

    # accumulate elements to get the CDF of the exponential distribution
    domain_cdf = np.cumsum(domain_pdf).tolist()
    # pick a uniformly random value on the CDF
    pick = np.random.uniform()

    # return the index corresponding to the pick
    # take the min between the index and  len(D)-1 to prevent returning index out of bound
    return domain[min(np.searchsorted(domain_cdf, pick) + 1, len(domain)-1)]


def a_dist(data, domain, quality_function, eps, delta, bulk=False):
    """A_dist algorithm
    :param data: database
    :param domain:
    :param quality_function: sensitivity-1 quality function
    :param eps: privacy parameter
    :param delta: privacy parameter
    :return: an element of domain with maximum value of quality function or 'bottom'
    """

    # compute q(X,i) for all the elements in D
    if bulk:
        qualified_domain = quality_function(data, domain)
    else:
        qualified_domain = [quality_function(data, i) for i in domain]
    h1_score = max(qualified_domain)
    h1 = domain[qualified_domain.index(h1_score)]  # h1 is domain element with highest quality
    qualified_domain.remove(h1_score)
    domain.remove(h1)
    h2_score = max(qualified_domain)
    # h2 = domain[qualified_domain.index(h2_score)]  # h2 is domain element with second-highest quality
    noisy_gap = h1_score - h2_score + np.random.laplace(0, 1 / eps, 1)
    if noisy_gap < np.log(1/delta)/eps:
        return 'bottom'
    else:
        return h1


def above_threshold_on_queries(data, queries, threshold, eps):
    initialized_threshold = above_threshold(data, threshold, eps)
    answers = []
    for q in queries:
        answers.append(initialized_threshold(q))
        if q == 'up':
            break
    return answers


def above_threshold(data, threshold, eps):
    """
    above_threshold algorithm - privacy preserving algorithm that given a stream of sensitivity-1 queries
    tests if their evaluation over the given data exceeds the threshold
    :param data: database
    :param threshold: fixed threshold
    :param eps: privacy parameter
    :return: threshold_instance that get queries as input
    and for every given query evaluate the private above-threshold test
    """

    noisy_threshold = threshold + np.random.laplace(0, 2 / eps, 1)

    def threshold_instance(query):
        noise = np.random.laplace(0, 4 / eps, 1)
        if query(data) + noise >= noisy_threshold:
            return 'up'
        else:
            return 'bottom'
    return threshold_instance


