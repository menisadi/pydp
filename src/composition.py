from math import sqrt, log, exp, factorial
from operator import mul
from itertools import compress

# TODO add documentation
# TODO need to be more readable

def nCr(n, r):
    return factorial(n) / factorial(r) / factorial(n-r)


def advanced(eps, delta, d_tag, k):
    e_tag = sqrt(2*k*log(1/d_tag))*eps + k*eps*(exp(eps)-1)
    return e_tag, k*delta+d_tag


def optimal_homogeneous(eps, delta, k):
    for i in xrange(k/2 + 1):
        sum_i = sum(nCr(k, l)*(exp((k-l)*eps)-exp((k-2*i+l)*eps)) for l in xrange(i))
        delta_i = sum_i/(1-exp(eps))**k
        yield (k-2*i)*eps, 1-(1-delta_i)*(1-delta)**k


# eps is homogeneous
# delta_0 > 0, for i>0 : delta_i = 0
# no formal proof
def __optimal_homogeneous_e__(eps, delta, k):
    for i in xrange(k/2 + 1):
        sum_i = sum(nCr(k, l)*(exp((k-l)*eps)-exp((k-2*i+l)*eps)) for l in xrange(i))
        delta_i = sum_i/(1-exp(eps))**k
        yield (k-2*i)*eps, 1-(1-delta_i)*(1-delta)


def delta_bound(ds):
    return 1-reduce(mul, [1-d for d in ds])


def __splitter__(k):
    l = range(1, k+1)
    d = k
    for i in range(2**k):
        b = ('{0:0%db}' % d).format(i)
        bb = [i == '1' for i in b]
        split = set(compress(l, bb))
        yield split, set(l)-split


def __part_es__(es, s):
    return sum(es[i-1] for i in s)


def optimal_hetrogeneous(es, ds, eg, dg):
    k = len(es)
    sum_eg = sum([max(0, exp(__part_es__(es, s[0]))-exp(eg)*exp(__part_es__(es, s[1])))
                  for s in __splitter__(k)])
    le = 1 / reduce(mul, [1 + exp(e) for e in es])
    ld = 1 - (1-dg)/(reduce(mul, [1-d for d in ds]))
    return le*sum_eg <= ld


def find_eg(es, ds, dg, t):
    k = len(es), len(es)
    i, eg, r_g = sum(es), sum(es), sum(es)
    while t > 0:
        i /= 2.0
        # print t, i , eg
        a = (-1)**optimal_hetrogeneous(es, ds, eg, dg)
        if a == -1:
            r_g = eg
        eg += a*i
        if eg > k:
            raise ValueError("delta_g too small")
        t -= 1
    return r_g
