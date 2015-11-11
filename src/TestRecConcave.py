import unittest
import rec_concave
import examples
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats


class TestRecConcave(unittest.TestCase):

    def exact_median_interval(self, data, int_max_range):
        qualities = examples.bulk_quality_minmax(data, range(0, int(int_max_range)))
        max_quality = max(qualities)
        first = qualities.index(max_quality)
        last = (len(qualities) - 1) - qualities[::-1].index(max_quality)
        return first, last

    def dist_bound(self, eps, delta, alpha, beta):
        """
        calculate the minimum sample size for which A_dist at step 9 will fail
        only in probability < beta
        :return: the minimum samples required for A_dist to run
        """
        promise = 4 * np.log(1 / (beta * delta)) / alpha / eps
        return 2 * promise

    def setUp(self):
        self.range_end = 1000  # 2**14 + 1

        self.alpha = 0.2
        self.eps = 0.5
        self.delta = 1/float(self.range_end)
        self.RECURSION_BOUND = 2

        self.samples_size = 50  # self.dist_bound(self.eps, self.delta, self.alpha, 0.1)

        data_center = np.random.uniform(self.range_end/3, self.range_end/3*2)
        self.data = examples.get_random_data(self.samples_size, pivot=data_center)
        self.data = sorted(filter(lambda x: 0 <= x <= self.range_end, self.data))
        qualities = examples.bulk_quality_minmax(self.data, range(self.range_end))
        self.maximum_quality = max(qualities)

        print "the exact median is: %d" % np.median(self.data)
        print "the best quality of a domain element: %d" % self.maximum_quality
        print "which lies within the range: %s" % (self.exact_median_interval(self.data, self.range_end),)

        # TODO temp - remove later
        # plot the samples
        norm_pdf = scipy.stats.norm(data_center, self.samples_size)
        xs = np.linspace(0, self.range_end, self.range_end)
        plt.plot(self.data, [norm_pdf.pdf(sample) for sample in self.data], 'ro', xs, norm_pdf.pdf(xs), 'b')
        plt.show()
        # plot the range qualities
        plt.plot(range(self.range_end), qualities, 'bo')
        plt.show()

    def test_rec_concave_basis_median(self):
        print "testing basis to find median"

        result_depth_1 = rec_concave.evaluate(self.data, self.range_end, examples.quality_minmax, self.maximum_quality,
                                    self.alpha, self.eps, self.delta, 1)
        print "result from rec_concave: %d" % result_depth_1
        result_quality = examples.quality_minmax(self.data, result_depth_1)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

    def test_rec_concave_median(self):
        print "testing depth-2 to find median"

        result_depth_2 = rec_concave.evaluate(self.data, self.range_end, examples.quality_minmax, self.maximum_quality,
                                    self.alpha, self.eps, self.delta, 2)
        print "result from rec_concave: %d" % result_depth_2
        result_quality = examples.quality_minmax(self.data, result_depth_2)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

if __name__ == '__main__':
    unittest.main()
