import scipy.stats
import numpy as np

from .entries import complex_gaussian

def random_wishart(M,B):
    """
    MxM à partir de matrices MxB (B échantillons de M dimensional time series)
    """
    X = complex_gaussian(np.zeros(M), np.identity(M), B)
    #return scipy.stats.wishart.rvs(df=B, scale=np.identity(M), size=(1,1))/(B+1) # problème renvoie des wishart réelles
    return X @ np.conj(X.T) / (B+1)



def marchenko_pastur(x, c):
    lambda_plus = (1+np.sqrt(c))**2
    lambda_moins = (1-np.sqrt(c))**2
    result =  np.sqrt((lambda_plus - x) * (x - lambda_moins)) / (2 * np.pi * c * x)
    result[x>lambda_plus] = 0
    result[x<lambda_moins] = 0
    return result



def equivalent_deterministe_logdet(B, c):
    return (B+1) * ((1-c) * np.log(1/(1-c)) - c)


def equivalent_deterministe_trace(M, c):
    """
    équivalent deterministe de ||S_cor(\nu) - I_M||_F^2
    ie Tr((S_cor(\nu) - I_M)(S_cor(\nu) - I_M)^*)
    = M * [(1+c) -2 +1] = M*c
    """
    return M*c
