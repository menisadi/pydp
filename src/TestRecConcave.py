import unittest
import rec_concave
import examples
import numpy as np


# testes are based on Example 3.8 on
# A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
class TestRecConcave(unittest.TestCase):

    def setUp(self):
        self.alpha = 0.2
        self.eps = 0.5
        self.delta = 0.01
        self.RECURSION_BOUND = 2

    def test_rec_concave_basis_threshold(self):
        log_data_size = 4
        data_range = 2**log_data_size
        sample_size = data_range*5
        data = examples.get_random_data(data_range, 'threshold')
        sample = examples.get_labeled_sample(data, sample_size)
        # print data
        thresh = data.index(0)
        print "the first 0 is in index: %d" % thresh
        # print "and his quality is: %d" % examples.interval_threshold_quality(sample, thresh)
        # print [examples.interval_threshold_quality(sample, i) for i in xrange(len(data))]

        result_depth_1 = rec_concave.evaluate(sample, data_range, examples.interval_threshold_quality, data_range,
                                    self.alpha, self.eps, self.delta, 1)
        print "result from rec_concave with basis run only: %d \n" % result_depth_1
        self.assertLessEqual(np.abs(result_depth_1-thresh), 3)

    def test_rec_concave_basis_median(self):
        data_center = np.random.uniform(50, 100)
        data = examples.get_random_data(1000, pivot=data_center)
        rounded_data = [int(x) for x in data]
        result_depth_1 = rec_concave.evaluate(rounded_data, max(rounded_data), examples.quality_minmax, -1,
                                    self.alpha, self.eps, self.delta, 1)
        print "the exact median is: %d" % np.median(rounded_data)
        print "result from rec_concave with basis run only: %d " % result_depth_1
        # print "result quality: %d \n" % examples.quality_minmax(rounded_data, result_depth_1)
        self.assertGreaterEqual(examples.quality_minmax(rounded_data, result_depth_1), -3)

    def test_rec_concave_median(self):
        data_center = np.random.uniform(50, 100)
        data = examples.get_random_data(100, pivot=data_center)
        rounded_data = [int(x) for x in data]
        result_depth_1 = rec_concave.evaluate(rounded_data, max(rounded_data), examples.quality_minmax, len(data)/2,
                                    self.alpha, self.eps, self.delta, 2)
        print "the exact median is: %d" % np.median(rounded_data)
        print "result from rec_concave with basis run only: %d " % result_depth_1
        # print "result quality: %d \n" % examples.quality_minmax(rounded_data, result_depth_1)
        self.assertGreaterEqual(examples.quality_minmax(rounded_data, result_depth_1), len(data)/2-3)

if __name__ == '__main__':
    unittest.main()
