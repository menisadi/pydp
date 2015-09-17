import unittest
import numpy as np
import basicdp


class test_basicdp(unittest.TestCase):
    domain_size = 1000
    data_size = 100
    eps = 0.500
    randD = []
    # print self.randD
    quality_median = lambda a,b:-abs(np.median(a)-b)

    def setup(self):
        self.randD = np.random.normal(1, domain_size, data_size)
        # print self.randD

    def teardown(self):
        self.randD = None

    #def test_noisy_max(self):
        # compare noisy_max mechanism for finding the minimum to the direct approach
        # i = basicdp.noisy_max(randD, range(1,domain_size), quality_median, eps)
        # A = sorted(map(quality_median, randD))
        # self.assertGreaterEqual(i, np.searchsorted(A, randD[i]))

    def test_exponential_mechanism(self) : 
        # Pr[quality_median(result) < quality_median(np.median(randD)-2/eps*(log(domain_size)+t)] < exp(-t)
        i = basicdp.exponential_mechanism(self.randD, range(1,self.domain_size), self.quality_median, self.eps)
        self.assertGreaterEqual(self.quality_median(i), self.quality_median(np.median(self.randD)-2/eps*(log(self.domain_size)+5)))



# q=lambda a,b:-abs(np.mean(a)-b)
# q = (lambda a,b:abs(max(a)-b))
# X=np.random.normal(100,10,50)
# exponential_mechanism(range(100),range(100),q,0.1)
# noisy_max(range(100),range(100),q,0.1)

if __name__ == '__main__':
    unittest.main()
