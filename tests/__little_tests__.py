from collections import Counter
from numpy.random import exponential, uniform, normal
import src.qualities as ql
import time
import src.basicdp as bdp
import src.examples as xp
import numpy as np
from src.good_radius import __neighbours__, __max_average_ball__
import matplotlib.pyplot as plt


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


def memoize_neighbours():
    data = np.random.normal(0, 5, (10, 2))
    r = 3
    mat = __neighbours__(data, r)
    print zip(*np.nonzero(mat))
    print [(list(data[i]), list(data[j])) for i, j in zip(*np.nonzero(mat))]
    plt.scatter(*zip(*data))
    found = False
    t = 0
    while not found and t < 10:
        i = np.random.randint(len(data))
        p = data[i]
        close_to_p = [list(data[j]) for j in xrange(len(data)) if mat[i, j] == 1]
        if close_to_p:
            found = True
        t += 1
    print i, p
    print mat
    return close_to_p


def max_average_ball():
    data = np.random.normal(0, 2.5, (10, 2))
    plt.scatter(*zip(*data))
    t, r = 5, 3
    mat = __neighbours__(data, r)
    print zip(*np.nonzero(mat))
    print [(list(data[i]), list(data[j])) for i, j in zip(*np.nonzero(mat))]
    print mat
    closest = np.sum(mat, axis=1).argsort()[t:]
    print closest
    print np.sum(mat[closest], axis=1)
    return __max_average_ball__(data, r, t)


def max_average_ball2():
    sample_number, k, r = 2 ** 10, 2, 4
    data_2d = np.random.randint(0, 500, (sample_number, 2))
    artificial_cluster_size = 2 ** 6
    artificial_cluster = np.random.randint(100, 130, (artificial_cluster_size, 2))
    data_2d = np.vstack((data_2d, artificial_cluster))
    sample_number += artificial_cluster_size
    mat = __neighbours__(data_2d, r)
    t = 50
    closest = np.sum(mat, axis=1).argsort()[t:]
    return __max_average_ball__(data_2d, r, t)


def run_all_tests():
    print test_point_count_intervals_bounding(1, False)
    print test_exponential_mechanism_sparse(bdp.exponential_mechanism_big, False)
    compare_maximum_in_interval_versions()
    compare_interval_creation()
    print noisy_avg(2, 10000, 0.5, 3**-10, 10)
    print memoize_neighbours()
    print max_average_ball2()


def run_selected_test():
    print max_average_ball2()
    return


start_time = time.time()
run_selected_test()
# plt.show()
print "run-time: %.2f seconds" % (time.time() - start_time)

