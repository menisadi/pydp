import numpy as np
import math

def noisy_max(X,D,q,eps) :
	"""Noisy-Max Mechanism 
	noisyMax ( data , domain, quality function , privacy parameter )
	"""
	# compute q(X,i) + Lap(1/eps) for all the elements in D
	noisy = [q(X,i)+np.random.laplace(0, 1/eps, 1) for i in D]
	# return element with maximum noisy q(X,i)
	return 	noisy.index(max(noisy))

def exponential_mechanism(X,D,q,eps) :
	"""Exponential Mechanism 
	expo ( data , domain , quality function , privacy parameter )
	"""
	# calculate a list of probabilities for each element in the domain D
	# probability of element d in D proportional to exp(eps*q(X,d)/2)
	D_PDF = [math.exp(eps*q(X,d)/2) for d in D] 
	normalizer = sum(D_PDF) 
	D_PDF = [d/normalizer for d in D_PDF]
	# accumulate elements to get the CDF of the exponential distribution
	D_CDF = np.cumsum(D_PDF).tolist()
	# use the uniform distribution (from 0 to 1) to pick an elements by the CDF
	pick = np.random.rand()
	return np.searchsorted(D_CDF,pick) + 1

