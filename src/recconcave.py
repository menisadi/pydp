import basicdp
import math


def reconcave_basis(T, q, eps, S):
    """recursion basis for the reconcave procedure - execute the exponential mechanism
    note that the parameters r,alpha,delta and N are not being used
    reconcave_basis(solution set size, quality function of sensitivity 1, eps privacy parameter, solution set)
    """
    return basicdp.exponential_mechanism(S, range(T + 1), q, eps)


def evaluate(T, q, r, alpha, eps, delta, S, N):
    if N == 1 or T <= 32:
        return reconcave_basis(T, q, r, alpha, eps, delta, S)
    else:
        N -= 1

    logT = math.ceil(math.log(T, 2))
    Tup = 2 ** logT

    # TODO change the variables names to something more readable
    def q_new(s, i):
        if T < i <= Tup:
            min(0, q(s, T))
        else:
            return q(s, i)

    # def L(s, j):
    #     if 0 <= j <= logT:
    #         return max(min(q_new(s,i)))

    return
