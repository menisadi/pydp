import numpy as np
import math

def noisy_max(X,q,eps) :
	"""Noisy-Max Mechanism 
	noisyMax ( data , quality function , privacy parameter )
	"""
	# adds Laplasian noise in scale - 1/eps to all the elemnts in X
	noisy = [q(i)+np.random.laplace(0, 1/eps, 1) for i in X]
	# returns the index of the maximal element from the noisy elemntss
	return 	noisy.index(max(noisy))

def expo(X,R,q,eps) :
	"""Exponential Mechanism 
	expo ( data , range , quality function , privacy parameter )
	"""
	# calculate a list of probabilities for eash elemnt in X (the)
	R_chances = [math.exp(eps*q(X,r)) for r in R] 
	normalizator = sum(R_chances) 
	R_chances = [r/normalizator for r in R_chances]
	# accumulates elements to get the CDF of the exponential ditribuition
	R_CDF = np.cumsum(R_chances).tolist()
	# use theunform distribuition (from 0 to 1) to pick an elemnts by the CDF
	pick = np.random.rand()
	return np.searchsorted(R_CDF,pick) + 1

