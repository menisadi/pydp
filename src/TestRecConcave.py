import unittest
import rec_concave
import examples
import numpy as np


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
        self.delta = 1e-4
        self.RECURSION_BOUND = 2

    def test_rec_concave_basis_median(self):
        print "testing basis to find median"
        data_center = np.random.uniform(50, 100)
        data = examples.get_random_data(1000, pivot=data_center)

        # rounded_data = [int(x) for x in data]
        max_element = int(np.ceil(max(data)))
        result_depth_1 = rec_concave.evaluate(data, max_element, examples.quality_minmax, -1,
                                    self.alpha, self.eps, self.delta, 1)
        maximum_quality = max(examples.bulk_quality_minmax(data, range(max_element)))
        print "the exact median is: %d" % np.median(data)
        print "the best quality of a domain element: %d" % maximum_quality
        print "result from rec_concave with basis run only: %d" % result_depth_1
        print "and its quality: %d \n" % examples.quality_minmax(data, result_depth_1)
        self.assertGreaterEqual(examples.quality_minmax(data, result_depth_1), -3)

    def test_rec_concave_median(self):
        print "testing depth-2 to find median"
        data_center = np.random.uniform(50, 100)
        data = examples.get_random_data(1000, pivot=data_center)
        # rounded_data = [int(x) for x in data]
        max_element = int(np.ceil(max(data)))
        result_depth_2 = rec_concave.evaluate(data, max_element, examples.quality_minmax, len(data)/2,
                                    self.alpha, self.eps, self.delta, 2)
        true_median = np.median(data)
        maximum_quality = max(examples.bulk_quality_minmax(data, range(max(data))))
        print "the exact median is: %d" % true_median
        print "the best quality of a domain element: %d" % maximum_quality
        print "which lies within the range: %s" % (self.exact_median_interval(data,max_element),)
        print "result from rec_concave: %d" % result_depth_2
        print "and its quality: %d \n" % examples.quality_minmax(data, result_depth_2)
        self.assertLessEqual(np.abs(result_depth_2-true_median), 3)

if __name__ == '__main__':
    unittest.main()
