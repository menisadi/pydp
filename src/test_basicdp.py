import unittest
import numpy as np
import basicdp

class TestBasicDP(unittest.TestCase) :

	def setup(self) :
		randD = np.random.randint(1,1000,50)
		eps = 0.100

	def half(x) :
		return x/2

	def teardown(self) :
		randD = None

	def test_noisy_max(self) :
		i = noisy_max(D,half,eps)
		A = sorted(map(half,D))
		self.assertGreaterEqual(i,np.searchsorted(A,D[i]))

if __name__ == '__main__':
	unittest.main()