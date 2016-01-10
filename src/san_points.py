from collections import Counter
from basicdp import choosing_mechanism
from examples import concept_query
from random import randint


def q(samples, x):
    return sum(1 for s in samples if s == x)


def sanitize(samples, alpha, beta, eps, delta):
    remaining_samples = Counter(samples)
    for i in range(int(alpha/2)):
        b = choosing_mechanism()
    return


d = [randint(1, 5) for i in xrange(10)]
print d


def point(i):
    def concept(x):
        if x == i:
            return 1
        else:
            return 0
    return concept

c = point(4)
print concept_query(d,c)
