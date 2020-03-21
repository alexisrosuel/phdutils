import scipy.stats
import numpy as np

from .entries import complex_gaussian



def equivalent_deterministe(M, c, test_type):
    """
    dans le cas du test FNT:
    équivalent deterministe de ||S_cor(\nu) - I_M||_F^2
    ie Tr((S_cor(\nu) - I_M)(S_cor(\nu) - I_M)^*)
    = M * [(1+c) -2 +1] = M*c
    """
    if test_type=='logdet':
        return -M*((c-1)/c*np.log(1-c)-1)
    elif test_type=='frobenius':
        return M*c



def MP(x,c):
    lambda_plus = (1+np.sqrt(c))**2
    lambda_moins = (1-np.sqrt(c))**2
    val = np.sqrt((x-lambda_moins)*(lambda_plus-x)) / (2*np.pi*x*c)
    val[val<=0] = 0
    return val 
