from collections import Counter
from basicdp import choosing_mechanism
from qualities import concept_query
from examples import point_concept
from numpy import log, sqrt, abs
from numpy.random import randint, laplace
from bounds import choosing_mechanism_data_size


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


def sanitize(samples, domain, alpha, beta, eps, delta):
    remaining_samples = domain  # Counter(samples)
    est = dict.fromkeys(samples, 0)
    new_beta = alpha * beta / 4
    new_eps = eps / sqrt(32 * log(5/delta) / alpha)
    new_delta = alpha * delta / 5
    for i in range(int(2/alpha)):
        q = __point_choice_quality__(remaining_samples)
        b = choosing_mechanism(samples, remaining_samples, q, 1, alpha/2, new_beta, new_eps, new_delta)
        if b != 'bottom':
            remaining_samples.remove(b)
            # remaining_samples[b] -= 1
            # remaining_samples += Counter()
            est[b] = concept_query(samples, point_concept(b)) + laplace(0, 1 / eps / len(samples), 1)[0]
    return est


a, b, e, d = 0.2, 0.1, 0.5, 2**-20
n_a = a / 2
n_b = a * b / 4
n_e = e / sqrt(32 * log(5/d) / a)
n_d = a * d / 5
m = int(choosing_mechanism_data_size(1, n_a, n_b, n_e, n_d)) + 1
n = 20
print m
d = [randint(1, n/2) for i in xrange(m/5)]
d.extend([randint(n/2, n) for k in xrange(4*m/5+1)])
print len(d)
print Counter(d)
san = sanitize(d, set(range(n)), 0.2, 0.1, 0.5, 2**-20)
print [enumerate((d, point_concept(z))) for z in xrange(1, n)]
print san
print sum([abs(concept_query(d, point_concept(z)) - san[z]) > a for z in xrange(1, n)])/float(n)

