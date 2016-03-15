from collections import Counter
from numpy.random import exponential
import src.qualities as ql
import time
import src.basicdp as bdp
import numpy as np


def test_point_count_intervals_bounding(both_versions=True):
    s = [int(i) for i in exponential(100, 1000000)]
    i = (3, 12)
    s = sorted(s)
    print s
    print Counter(s)
    print ql.points_in_subset(s, i)
    start_time = time.time()
    print ql.point_count_intervals_bounding(s, i, 2)
    mid_time = time.time()
    print "first run-time: %.2f seconds" % (mid_time - start_time)
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


# print test_point_count_intervals_bounding(False)
print test_exponential_mechanism_sparse(bdp.exponential_mechanism_big, False)
