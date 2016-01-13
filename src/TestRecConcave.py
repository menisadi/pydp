import unittest
import rec_concave
import examples
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import bounds
import qualities


class TestRecConcave(unittest.TestCase):

    def exact_median_interval(self, data, int_max_range):
        bulk_qualities = qualities.bulk_quality_minmax(data, range(0, int(int_max_range)))
        max_quality = max(bulk_qualities)
        first = bulk_qualities.index(max_quality)
        last = (len(bulk_qualities) - 1) - bulk_qualities[::-1].index(max_quality)
        return first, last

    def setUp(self):
        self.range_end = 2**14

        self.alpha = 0.2
        self.eps = 0.5
        self.delta = 1/float(self.range_end)
        self.RECURSION_BOUND = 2

        self.samples_size = bounds.dist_bound(self.eps, self.delta, self.alpha, 0.01)
        print "range size: %d" % self.range_end
        print "sample size: %d" % self.samples_size
        data_center = np.random.uniform(self.range_end/3, self.range_end/3*2)
        self.data = examples.get_random_data(self.samples_size, pivot=data_center)
        self.data = sorted(filter(lambda x: 0 <= x <= self.range_end, self.data))
        bulk_qualities = qualities.bulk_quality_minmax(self.data, range(self.range_end))
        self.maximum_quality = max(bulk_qualities)

        print "the exact median is: %d" % np.median(self.data)
        print "the best quality of a domain element: %d" % self.maximum_quality
        print "which lies within the range: %s" % (self.exact_median_interval(self.data, self.range_end),)

        # TODO temp - remove later
        # plot the samples
        norm_pdf = scipy.stats.norm(data_center, self.samples_size / 6)
        xs = np.linspace(0, self.range_end, self.range_end)
        plt.plot(self.data, [norm_pdf.pdf(sample) for sample in self.data], 'ro', xs, norm_pdf.pdf(xs), 'b')
        plt.show()
        # plot the range qualities
        plt.plot(range(self.range_end), qualities, 'bo')
        plt.show()

    def test_rec_concave_basis_median(self):
        print "testing basis to find median"

        result_depth_1 = rec_concave.evaluate(self.data, self.range_end, qualities.bulk_quality_minmax, self.maximum_quality,
                                    self.alpha, self.eps, self.delta, 1, True)
        print "result from rec_concave: %d" % result_depth_1
        result_quality = qualities.quality_minmax(self.data, result_depth_1)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

    def test_rec_concave_median(self):
        print "testing depth-2 to find median"

        result_depth_2 = rec_concave.evaluate(self.data, self.range_end, qualities.bulk_quality_minmax,
                                              self.maximum_quality, self.alpha, self.eps, self.delta, 2, True)
        print "result from rec_concave: %d" % result_depth_2
        result_quality = qualities.quality_minmax(self.data, result_depth_2)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

if __name__ == '__main__':
    unittest.main()