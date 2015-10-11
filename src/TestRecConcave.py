import unittest
import numpy as np
import random


# testes are based on Example 3.8 on
# A. Beimel, K. Nissim, and U. Stemmer. Private learning and sanitization
class TestRecConcave(unittest.TestCase):
    def interval_threshold_quality(self, sampled_data, threshold_index):
        # assuming that sampled_data is two list of the same length - one of x's and one of y's
        xs = sampled_data[0]
        ys = sampled_data[1]
        # sum the number of indexes which 'agree' to the give threshold
        return sum([(ys[i] == 0 and xs[i] > threshold_index) or
                    (ys[i] == 1 and xs[i] < threshold_index) for i in xrange(len(xs))])

    def setUp(self):
        self.LOG_DATA_SIZE = 10
        self.DATA_RANGE = 2**self.LOG_DATA_SIZE
        self.THRESHOLD = np.random.randint(self.DATA_RANGE)
        self.DATA = [1]*(self.THRESHOLD - 1) + [0]*(self.DATA_RANGE - self.THRESHOLD + 1)
        self.SAMPLE_XS = sorted(random.sample(xrange(self.DATA_RANGE), self.LOG_DATA_SIZE))
        self.SAMPLE_YS = [self.DATA[self.SAMPLE_XS[i]] for i in xrange(len(self.SAMPLE_XS))]
        # TODO should we test on a 'true' fixed threshold or a sampled one like this?
        self.SAMPLE_THRESHOLD_MAX = self.SAMPLE_XS[self.SAMPLE_YS(0)] - 1
        self.SAMPLE_THRESHOLD_MIN = self.SAMPLE_XS[self.SAMPLE_YS(0)-1]+1
        self.SAMPLE = [self.SAMPLE_XS, self.SAMPLE_YS]

    def test_rec_concave_basis(self):
        self.assertEqual(True, False)

    def test_rec_concave_utility(self):
        self.assertEqual(True, False)

if __name__ == '__main__':
    unittest.main()
