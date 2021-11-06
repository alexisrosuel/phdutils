import numpy as np
import scipy.integrate

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



def complex_integration(f, a, b):
    return scipy.integrate.quad(lambda x: np.real(f(x)), a=a, b=b)[0] + 1j*scipy.integrate.quad(lambda x: np.imag(f(x)), a=a, b=b)[0]




def contour_integral(f, R):
    """
    Compute the contour integral of f over the circle C(0,R). The integral is normalized by 2i pi
    """
    cont_int = complex_integration(lambda theta: f(R*np.exp(1j*theta)) * 1j*R*np.exp(1j*theta), a=0, b=2*np.pi)
    return cont_int / (2*np.pi*1j)


###### Stieltjes

def inversion_stieltjes(x_range, s):
    """
    Compute the absolutely continuous density part of the stieltjes transform s over x_range.
    """
    y=0.000001
    return np.array([np.imag(s(x+1j*y)) for x in x_range])/np.pi
