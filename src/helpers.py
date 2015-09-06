import numpy as np
import math

def noisyMax(X,q,eps) :
	noisy = [q(i)+np.random.laplace(0, 1/eps, 1) for i in X]
	return 	noisy.index(max(noisy))

def exponential(X,q,eps) :
	norm += (math.exp(eps*q(X,r))