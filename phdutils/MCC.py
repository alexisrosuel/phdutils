import numpy as np


def compute_MCC(C_hats, mask):
    '''
    Compute the MCC (scalar) from C_hats (length N) with specified mask
    '''
    max_provisoire = []
    for C_hat in C_hats:
        C_hat_square = (np.abs(C_hat)**2).astype(np.float)
        np.fill_diagonal(C_hat_square, -np.inf)
        max_C_square = np.max(C_hat_square)
        max_provisoire.append(float(max_C_square))
    max_provisoire = np.array(max_provisoire)*mask
    return np.max(max_provisoire)




def scale_MCC(N, B, M, MCC):
    an = -np.log(N/(B+1)) -np.log(M*(M-1)/2)
    MCC_scaled = (B+1)*MCC + an
    return MCC_scaled



def create_mask(N, B):
    mask = np.zeros(N)
    for i in range(int(N/(B+1))):
        mask[i*(B+1)] = 1
    return mask


def compute_MCCs_from_C_hats_sample(C_hats_sample, B, N):
    '''
    Return the MCC for each time series
    '''
    MCCs = []
    mask = create_mask(N=N, B=B)
    for C_hats in C_hats_sample:
        MCC = compute_MCC(C_hats=C_hats, mask=mask)
        MCCs.append(MCC)
    return np.array(MCCs)


def compute_C_hats_sample_from_Ys(Ys, B):
    '''
    Compute C_hats for each repeat of the time series
    '''
    C_hats_sample = []
    for repeat in range(Ys.shape[0]):
        Y = Ys[repeat,:,:]
        C_hats = compute_C_hats(Y=Y, B=B)
        C_hats_sample.append(C_hats)
    return C_hats_sample


def compute_MCCs_from_Ys(Ys, B):
    '''
    Compute C_hats for each repeat of the time series
    '''
    nb_repeat, M, N = Ys.shape
    mask = create_mask(N=N, B=B)

    MCCs = []
    for repeat in range(nb_repeat):
        Y = Ys[repeat,:,:]
        C_hats = compute_C_hats(Y=Y, B=B)
        MCC = compute_MCC(C_hats=C_hats, mask=mask)
        MCCs.append(MCC)
    return np.array(MCCs)




def compute_MCCs_from_Ys_2(Ys, B):
    N = Ys.shape[-1]
    C_hats_repeats = compute_C_hats_sample_from_Ys(Ys=Ys, B=B)
    MCCs = compute_MCCs_from_C_hats_sample(C_hats_sample=C_hats_sample, B=B, N=N)
    return MCCs
