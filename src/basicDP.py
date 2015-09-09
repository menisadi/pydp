import numpy as np
import math

# Noisy-Max Mechanism
# 
def noisyMax(X,q,eps) :
	# adds Laplasian noise in scale - 1/eps to all the elemnts in X
	noisy = [q(i)+np.random.laplace(0, 1/eps, 1) for i in X]
	# returns the index of the maximal element from the noisy elemntss
	return 	noisy.index(max(noisy))

# Exponential Mechanism
# 
def expo(X,q,eps) :
	# calculate a list of probabilities for eash elemnt in X (the)
	X_chances = [math.exp(eps*q(X,r)) for r in X] 
	normalizator = sum(X_chances) 
	X_chances = [i/normalizator for i in X_chances]
	# accumulates elements to get the CDF of the exponential ditribuition
	X_CDF = np.cumsum(X_chances).tolist()
	# use theunform distribuition (from 0 to 1) to pick an elemnts by the CDF
	pick = np.random.rand()
	return np.searchsorted(X_CDF,pick) + 1

