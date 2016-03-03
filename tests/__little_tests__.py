from collections import Counter
from numpy.random import exponential
import src.qualities as ql
import time


def test_point_count_intervals_bounding(both=True):
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
    if both:
        print ql.point_count_intervals_bounding2(s, i, 2)
        print "second run-time: %.2f seconds" % (time.time() - mid_time)


test_point_count_intervals_bounding(False)
