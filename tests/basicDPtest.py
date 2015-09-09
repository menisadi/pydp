import unitetest
import numpy as np
import ../src/basicDP

randD = np.random.randint(1,100,50)
eps = 0.100
def id(x) :
	return x

class TestBasicDP(unitetest.TestCase) :
	def test_noisy_max(self) :
		i = noisyMax(D,id,eps)
		

