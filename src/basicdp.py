import numpy as np
import math


def noisy_max(X, D, q, eps):
    """Noisy-Max Mechanism
    noisy_max ( data , domain, quality function , privacy parameter )
    """

    # compute q(X,i) + Lap(1/eps) for all the elements in D
    noisy = [q(X, i) + np.random.laplace(0, 1 / eps, 1) for i in D]
    # return element with maximum noisy q(X,i)
    return noisy.index(max(noisy))


def exponential_mechanism(X, D, q, eps):
    """Exponential Mechanism
    exponential_mechanism ( data , domain , quality function , privacy parameter )
    """

    # calculate a list of probabilities for each element in the domain D
    # probability of element d in D proportional to exp(eps*q(X,d)/2)
    D_PDF = [math.exp(eps * q(X, d) / 2) for d in D]
    normalizer = sum(D_PDF)
    D_PDF = [d / normalizer for d in D_PDF]
    normalizer = sum(D_PDF)
    # for debugging and other reasons: check that D_CDF indeed defines a distribution
    # use the uniform distribution (from 0 to 1) to pick an elements by the CDF
    if abs(normalizer - 1) > 0.001:
        raise ValueError('ERR: exponential_mechanism, sum(D_PDF) != 1.')

    # accumulate elements to get the CDF of the exponential distribution
    D_CDF = np.cumsum(D_PDF).tolist()
    # pick a uniformly random value on the CDF
    pick = np.random.rand()

    # return the index corresponding to the pick
    # take the min between the index and  len(D)-1 to prevent returning index out of bound
    return min(np.searchsorted(D_CDF, pick) + 1, len(D)-1)


def a_dist(eps, delta, data, quality_function):
    """A_dist algorithm
    :param eps, delta: privacy parameters
    :param data: database
    :param quality_function: sensitivity-1 quality function
    :return:
    """
    qualified_data = [quality_function(x) for x in data]
    two_highest_scores_indexes = np.argpartition(np.array(qualified_data), -2)[-2:]
    h1 = max(qualified_data[two_highest_scores_indexes])
    h2 = min(qualified_data[two_highest_scores_indexes])
    noisy_gap = h1 - h2 + np.random.laplace(0, 1 / eps, 1)
    # TODO should it be an error or just return?
    # TODO change the error message
    if noisy_gap < math.log(1/delta)/eps:
        raise ValueError('ERR: The gap between the two highest scores is two small')
    return h1

