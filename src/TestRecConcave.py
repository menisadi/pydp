import unittest
import rec_concave
import examples
import numpy as np

# testes are based on Example 3.8 on
# A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
class TestRecConcave(unittest.TestCase):
    # TODO choose approach
    # first approach
    def interval_threshold_quality(self, sampled_data, threshold_index):
        # assuming that sampled_data is two list of the same length - one of x's and one of y's
        xs = sampled_data[0]
        ys = sampled_data[1]
        # sum the number of indexes which 'agree' to the give threshold
        return sum([(ys[i] == 0 and xs[i] >= threshold_index) or
                    (ys[i] == 1 and xs[i] < threshold_index) for i in xrange(len(xs))])

    # second approach
    def threshold_function(self, index, threshold):
        if index < threshold:
            return 1
        else:
            return 0

    # second approach
    def interval_threshold_quality2(self, sampled_data, threshold_index):
        def concept(x):
            return self.threshold_function(x, threshold_index)
        return examples.concept_quality(sampled_data, concept)

    def setUp(self):
        self.LOG_DATA_SIZE = 4
        self.DATA_RANGE = 2**self.LOG_DATA_SIZE
        self.SAMPLE_SIZE = self.DATA_RANGE*5
        self.DATA = examples.get_random_data(self.DATA_RANGE, 'threshold')
        # TODO should we test on a 'true' fixed threshold or a sampled one like this?
        self.SAMPLE = examples.get_labeled_sample(self.DATA, self.SAMPLE_SIZE)
        # TODO calculate the limits of 'good' threshold
        # self.SAMPLE_THRESHOLD_MAX = self.SAMPLE[0][self.SAMPLE[1][0]] - 1
        # self.SAMPLE_THRESHOLD_MIN = self.SAMPLE_XS[self.SAMPLE[1][0]-1] + 1

        self.alpha = 0.2
        self.EPS = 0.5
        self.delta = 0.01
        self.RECURSION_BOUND = 2

    def test_rec_concave_basis(self):
        print self.DATA
        thresh = self.DATA.index(0)
        print "the first 0 is in index: %d" % thresh
        print "and his quality is: %d" % self.interval_threshold_quality(self.SAMPLE, thresh)
        print [self.interval_threshold_quality(self.SAMPLE, i) for i in xrange(len(self.DATA))]

        result_depth_1 = rec_concave.evaluate(self.DATA_RANGE, self.interval_threshold_quality,
                                    self.DATA_RANGE,
                                    self.alpha, self.EPS, self.delta,
                                    self.SAMPLE, 1)
        print "result from rec_concave with basis run only: %d " % result_depth_1
        self.assertLessEqual(np.abs(result_depth_1-thresh), 3)
"""
    def test_rec_concave_utility(self):
        result_depth_2 = rec_concave.evaluate(self.DATA_RANGE, self.interval_threshold_quality,
                                    self.DATA_RANGE,
                                    self.alpha, self.EPS, self.delta,
                                    self.SAMPLE, 2)
        print result_depth_2

        self.assertEqual(True, True)
"""
if __name__ == '__main__':
    unittest.main()
