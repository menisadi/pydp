from collections import Counter
from numpy.random import exponential, uniform, normal
import src.qualities as ql
import time
import src.basicdp as bdp
import src.examples as xp
import numpy as np


def test_point_count_intervals_bounding(j=2, both_versions=True):
    s = [int(i) for i in normal(100, 10, 1000000)]
    m = min(s)
    i = (m, m + 20)
    s = sorted(s)
    # print s
    print Counter(s)
    print "number of points in %s = %d" % (str(i), ql.points_in_subset(s, i))
    start_time = time.time()
    result = ql.point_count_intervals_bounding(s, i, j)
    print "maximum point in a sub-interval of %s with length 2^%d is %d" % (str(i), j, result)
    mid_time = time.time()
    print "run-time: %.2f seconds" % (mid_time - start_time)
    if both_versions:
        print ql.point_count_intervals_bounding2(s, i, 2)
        print "second run-time: %.2f seconds" % (time.time() - mid_time)


def test_pick_out_of_sub_group(tries=10):
    t = xrange(2**30)
    tp = range(2000)+range(2**20, 2**21)
    assert all([bdp.__pick_out_of_sub_group__(t, tp) not in tp for _ in xrange(tries)])


def test_exponential_mechanism_sparse(mechanism, test_pick=True):
    if test_pick:
        test_pick_out_of_sub_group()
    d = np.random.normal(0, 1, 1000)
    r = xrange(-2**30, 2**30)
    ri = range(-4, 4)
    # print ri
    result = bdp.sparse_domain(mechanism, d, r, ri, ql.quality_minmax, 0.1)
    exact_median = np.median(d)
    print "true median = %f" % exact_median
    print "and its quality is : %d" % ql.quality_minmax(d, exact_median)
    print "result = %f" % result
    print "and its quality is : %d" % ql.quality_minmax(d, result)


def compare_maximum_in_interval_versions():
    range_end = 2**40

    samples_size = uniform(1000,2000)
    print "range size: %d" % range_end
    print "sample size: %d" % samples_size
    data_center = np.random.uniform(range_end/3, range_end/3*2)
    data = xp.get_random_data(samples_size, pivot=data_center)
    data = sorted(filter(lambda x: 0 <= x <= range_end, data))

    maximum_quality1 = ql.min_max_maximum_quality(data, 0, range_end)
    maximum_quality2 = ql.__old_min_max_maximum_quality__(data, (0, range_end))
    print maximum_quality1 == maximum_quality2


def compare_interval_creation():
    range_end = 2**40

    samples_size = uniform(1000,2000)
    print "range size: %d" % range_end
    print "sample size: %d" % samples_size
    data_center = np.random.uniform(range_end/3, range_end/3*2)
    data = xp.get_random_data(samples_size, pivot=data_center)
    data = sorted(filter(lambda x: 0 <= x <= range_end, data))

    interval_length = 1
    range_start = 0
    old_list = xp.__old_build_intervals_set__(data, interval_length, range_start, range_end)
    new_list = xp.__build_intervals_set__(data, interval_length, range_start, range_end)
    assert len(old_list), len(new_list)
    print all(i[0] == j for i, j in zip(old_list, new_list))


def vec_avg(vs,q):
    st = set(map(tuple,vs))
    sm = sum(1 for v in st if q(v))
    return sum(v for v in vs if q(v))/float(sm)


def noisy_avg(dim, n, eps, delta, gs):
    s = np.random.randint(0, gs, (n, dim))
    p = lambda v: np.linalg.norm(v) <= gs
    res = bdp.noisy_avg(s, p, gs, dim, eps, delta)
    tr = vec_avg(s, p)
    dist = np.linalg.norm(res - tr)
    return dist / np.linalg.norm(tr)


# print test_point_count_intervals_bounding(1, False)
# print test_exponential_mechanism_sparse(bdp.exponential_mechanism_big, False)
# compare_maximum_in_interval_versions()
# compare_interval_creation()
print noisy_avg(2, 10000, 0.5, 3**-10, 10)
