import numpy as np
import math

def noisyMax(X,q,eps) :
	noisy = [q(i)+np.random.laplace(0, 1/eps, 1) for i in X]
	return 	noisy.index(max(noisy))

def expo(X,q,eps) :
	X_chances = [math.exp(eps*q(X,r)) for r in X]
	normalizator = sum(X_chances)
	X_chances = [i/normalizator for i in X_chances]
	X_CDF = np.cumsum(X_chances).tolist()
	pick = np.random.rand()
	return np.searchsorted(X_CDF,pick) + 1

