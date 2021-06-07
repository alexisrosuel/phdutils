import numpy as np
import scipy.stats


from .spectral_estimators import compute_C_hats

# MCC : Maximum Cross Coherency
# MCCT : Maximum Cross Coherency Test


def compute_MCC(C_hats):
    '''
    Compute the MCC (scalar) over all C_hats (length N)
    '''
    max_provisoire = []
    for C_hat in C_hats:
        C_hat_square = (np.abs(C_hat)**2).astype(np.float)
        np.fill_diagonal(C_hat_square, -np.inf)
        max_C_square = np.max(C_hat_square)
        max_provisoire.append(float(max_C_square))
    max_provisoire = np.array(max_provisoire)
    return np.max(max_provisoire)



def scale_MCC(MCC, M, B, N, nu_fixed=False):
    """
    Compute the normalization of the MCC given M and the number of frequencies considered.
    """
    mask = create_mask(N=N, B=B, nu_fixed=nu_fixed)
    an = -np.log(np.sum(mask)) -np.log(M*(M-1)/2)
    MCC_scaled = (B+1)*MCC + an
    return MCC_scaled



def create_mask(N, B, nu_fixed=False):
    mask = np.zeros(N)
    mask[0] = 1

    if not nu_fixed:
        i = 0
        while i*(B+1) < N-B:
            mask[i*(B+1)]=1
            i = i+1

    return mask



def get_GN(N, B):
    """
    GN is the subset of the Fourier frequencies k/N, where two consecutive elements are separated by B elements
    """
    mask = create_mask(N=N, B=B)
    GN = np.linspace(-0.5,0.5, N) * mask
    return GN[mask.astype(bool)]



####### Tests #######

def compute_MCCT(MCC, N, B, M, alpha_T):
    q_1_moins_alpha = scipy.stats.gumbel_r().ppf(1-alpha_T)
    return scale_MCC(B=B, M=M, N=N, MCC=MCC, nu_fixed=False) > q_1_moins_alpha



def compute_WZT(MCC, N, B, alpha_T):
    q_1_moins_alpha = scipy.stats.gumbel_r().ppf(1-alpha_T)
    return scale_MCC(B=B, M=2, N=N, MCC=MCC, nu_fixed=False) > q_1_moins_alpha







############# "Shortcut functions" #############


def compute_MCC_from_C_hats_sample(C_hats_sample):
    '''
    Return the MCC for each time series
    '''
    MCCs = []

    for C_hats in C_hats_sample:
        MCC = compute_MCC(C_hats=C_hats)
        MCCs.append(MCC)
    return np.array(MCCs)
