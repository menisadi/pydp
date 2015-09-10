import basicdp
import numpy as np
import math

def evaluate(T,q,r,alpha,eps,delta,S,N) :
	if N == 1 or T <= 32 : 
		return basicdp.expo(S, range(T+1), q, eps)
	else :
		N = N - 1

	Tup = 2**math.ceil(math.log(T,2))
	min(0,q(S,T))

