import numpy as np

def get_M(c=None, alpha=None, B=None, N=None):
    if B is not None:
        return np.around((B+1)*c).astype(int)
    elif N is not None:
        return np.around(N**alpha).astype(int)

def get_N(alpha=None, c=None, B=None, M=None):
    if B is not None:
        return np.around((c*(B+1))**(1/alpha)).astype(int)
    elif M is not None:
        return np.around(M**(1/alpha)).astype(int)

def get_B(alpha=None, c=None, N=None, M=None):
    if N is not None:
        return np.around((N**alpha)/c).astype(int)
    elif M is not None:
        return np.around(M/c).astype(int)



def compute_repartition(sample, x_range):
    """
    Return the empirical cummulative distribution given a s sample
    """
    repartition = []
    for x in x_range:
        repartition.append(np.mean(sample<x))
    return np.array(repartition)
