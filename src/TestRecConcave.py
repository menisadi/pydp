import unittest
import rec_concave
import examples
import numpy as np
import matplotlib.pyplot as plt


class TestRecConcave(unittest.TestCase):

    def exact_median_interval(self, data, int_max_range):
        qualities = examples.bulk_quality_minmax(data, range(0, int(int_max_range)))
        max_quality = max(qualities)
        first = qualities.index(max_quality)
        last = (len(qualities) - 1) - qualities[::-1].index(max_quality)
        return first, last

    def setUp(self):
        self.alpha = 0.2
        self.eps = 0.5
        self.delta = 1e-2
        self.RECURSION_BOUND = 2

        self.range_end = 200 + 1
        self.samples_size = 70

        data_center = np.random.uniform(0, self.range_end)
        self.data = examples.get_random_data(self.samples_size, pivot=data_center)
        self.data = [i % self.range_end for i in self.data]

        qualities = examples.bulk_quality_minmax(self.data, range(self.range_end))
        self.maximum_quality = max(qualities)

        print "the exact median is: %d" % np.median(self.data)
        print "the best quality of a domain element: %d" % self.maximum_quality
        print "which lies within the range: %s" % (self.exact_median_interval(self.data, self.range_end),)

        plt.plot(range(self.range_end), qualities, 'bo')
        plt.show()

    def test_rec_concave_basis_median(self):
        print "testing basis to find median"

        result_depth_1 = rec_concave.evaluate(self.data, self.range_end, examples.quality_minmax, self.maximum_quality,
                                    self.alpha, self.eps, self.delta, 1)
        if type(result_depth_1) == int:
            print "result from rec_concave: %d" % result_depth_1
        else:
            raise ValueError('stability problem')
        result_quality = examples.quality_minmax(self.data, result_depth_1)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

    def test_rec_concave_median(self):
        print "testing depth-2 to find median"

        result_depth_2 = rec_concave.evaluate(self.data, self.range_end, examples.quality_minmax, self.maximum_quality,
                                    self.alpha, self.eps, self.delta, 2)
        if type(result_depth_2) == int:
            print "result from rec_concave: %d" % result_depth_2
        else:
            raise ValueError('stability problem')
        result_quality = examples.quality_minmax(self.data, result_depth_2)
        print "and its quality: %d \n" % result_quality
        self.assertLessEqual(np.abs(result_quality - self.maximum_quality), 10)

if __name__ == '__main__':
    unittest.main()
