from basicdp import choosing_mechanism, choosing_mechanism_big
from qualities import concept_query
from examples import point_concept
from numpy import log, log2, sqrt, ceil
from numpy.random import laplace


def __point_choice_quality__(subset):
    def quality(samples, x):
        if x in subset:
            return sum(1 for s in samples if s == x)
        else:
            return 0
    return quality


def __counter_to_list__(counter):
    c_list = []
    for k in counter.keys():
        c_list.extend([k]*counter[k])
    return c_list


def sanitize(samples, alpha, beta, eps, delta):
    """

    :param samples:
    :param alpha:
    :param beta:
    :param eps:
    :param delta:
    :return:
    """
    max_sample = max(samples)
    dim = ceil(log2(max_sample + 1))
    end_domain = 2**int(dim)
    remaining_samples = set(range(end_domain))
    est = dict.fromkeys(remaining_samples, 0)
    new_beta = alpha * beta / 4
    new_eps = eps / sqrt(32 * log(5/delta) / alpha)
    new_delta = alpha * delta / 5
    for i in range(int(2/alpha)):
        q = __point_choice_quality__(remaining_samples)
        b = choosing_mechanism_big(samples, remaining_samples, q, 1, alpha/2, new_beta, new_eps, new_delta)
        if b != 'bottom':
            remaining_samples.remove(b)
            # remaining_samples[b] -= 1
            # remaining_samples += Counter()
            est[b] = concept_query(samples, point_concept(b)) + laplace(0, 1 / eps / len(samples), 1)[0]
    return est

