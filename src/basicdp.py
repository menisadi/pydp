import numpy as np
import math

def noisy_max(X,R,q,eps) :
	"""Noisy-Max Mechanism 
	noisyMax ( data , quality function , privacy parameter )
	"""
	# adds Laplace noise in scale - 1/eps to all the elements in X
	noisy = [q(X,i)+np.random.laplace(0, 1/eps, 1) for i in R]
	# returns the index of the maximal element from the noisy elements
	return 	noisy.index(max(noisy))

def exponential_mechanism(X,R,q,eps) :
	"""Exponential Mechanism 
	expo ( data , range , quality function , privacy parameter )
	"""
	# calculate a list of probabilities for each element in X (the)
	R_chances = [math.exp(eps*q(X,r)) for r in R] 
	normalizator = sum(R_chances) 
	R_chances = [r/normalizator for r in R_chances]
	# accumulates elements to get the CDF of the exponential distribution
	R_CDF = np.cumsum(R_chances).tolist()
	# use the uniform distribution (from 0 to 1) to pick an elements by the CDF
	pick = np.random.rand()
	return np.searchsorted(R_CDF,pick) + 1

