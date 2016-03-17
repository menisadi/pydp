from collections import Counter
from numpy.random import exponential, uniform
import src.qualities as ql
import time
import src.basicdp as bdp
import src.examples as xp
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


# print test_point_count_intervals_bounding(False)
# print test_exponential_mechanism_sparse(bdp.exponential_mechanism_big, False)
# compare_maximum_in_interval_versions()
compare_interval_creation()
