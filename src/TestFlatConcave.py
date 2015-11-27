import unittest
import flat_concave
import examples
import numpy as np
import bounds


class TestRecConcave(unittest.TestCase):

    def exact_median_interval(self, data, int_max_range):
        qualities = examples.bulk_quality_minmax(data, range(0, int(int_max_range)))
        max_quality = max(qualities)
        first = qualities.index(max_quality)
        last = (len(qualities) - 1) - qualities[::-1].index(max_quality)
        return first, last

    def setUp(self):
        self.range_end = 2**20

        self.alpha = 0.2
        self.eps = 0.5
        self.delta = 1/float(self.range_end)

        self.samples_size = bounds.dist_bound(self.eps, self.delta, self.alpha, 0.01)
        print "range size: %d" % self.range_end
        print "sample size: %d" % self.samples_size
        data_center = np.random.uniform(self.range_end/3, self.range_end/3*2)
        self.data = examples.get_random_data(self.samples_size, pivot=data_center)
        self.data = sorted(filter(lambda x: 0 <= x <= self.range_end, self.data))
        # self.data = self.data[np.sort(np.where((self.data >= 0) & (self.data < self.range_end)))]
        self.maximum_quality = examples.min_max_maximum_quality(self.data, (0, self.range_end))

        print "the exact median is: %d" % np.median(self.data)
        print "the best quality of a domain element: %d" % self.maximum_quality
        # print "which lies within the range: %s" % (self.exact_median_interval(self.data, self.range_end),)

    def test_flat_concave_median(self):
        print "testing flat_concave to find median"

        result = flat_concave.evaluate(self.data, self.range_end, examples.quality_minmax,
                                               self.maximum_quality, self.alpha, self.eps, self.delta,
                                               examples.min_max_intervals_bounding, examples.min_max_maximum_quality)
        print "result from flat_concave: %d" % result
        result_quality = examples.quality_minmax(self.data, result)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

if __name__ == '__main__':
    unittest.main()