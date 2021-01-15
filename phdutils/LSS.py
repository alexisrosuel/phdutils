import numpy as np

from .spectral_estimators import compute_C_hats
from .random_matrices import equivalent_deterministe

# LSS : Linear Spectral Statistics
# LSST : Linear Spectral Statistics Test

def compute_LSS(C_hats, c_N, test_type):
    """
    Compute the specified LSS for all frequencies
    """
    M = C_hats[0].shape[0]

    if test_type=='logdet':
        LSSs = [np.product(np.linalg.slogdet(C_hat)) for C_hat in C_hats]
        LSSs = -np.real(LSSs)

    elif test_type=='frobenius':
        LSSs = [np.linalg.norm(C_hat-np.identity(M))**2 for C_hat in C_hats]

    else:
        raise ValueError('test_type not recognised: %s' % test_type)

    #renormalisation
    LSSs = np.array(LSSs)/M
    return LSSs



def compute_LSST(C_hats, c_N, test_type):
    """
    Compute the test statistics associated with the LSS
    """
    LSSs = compute_LSS(C_hats, c_N, test_type)
    return np.max(np.abs(LSSs - equivalent_deterministe(c=c_N, test_type=test_type)))







############# "Shortcut functions" #############

def compute_LSST_from_C_hats_sample(C_hats_sample, c_N, test_type):
    LSSTs = []
    for C_hats in C_hats_sample:
        LSST = compute_LSST(C_hats=C_hats, c_N=c_N, test_type=test_type)
        LSSTs.append(LSST)
    return np.array(LSSTs)
