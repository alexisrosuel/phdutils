import scipy.stats
import numpy as np

from .entries import complex_gaussian



def equivalent_deterministe_logdet(B, c):
    return (B+1) * ((1-c) * np.log(1/(1-c)) - c)


def equivalent_deterministe_trace(M, c):
    """
    équivalent deterministe de ||S_cor(\nu) - I_M||_F^2
    ie Tr((S_cor(\nu) - I_M)(S_cor(\nu) - I_M)^*)
    = M * [(1+c) -2 +1] = M*c
    """
    return M*c
