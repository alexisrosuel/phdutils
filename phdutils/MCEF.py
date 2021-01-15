import numpy as np

from .spectral_estimators import compute_C_hats

# MCEF : Maximum Coherency Eigenvalue over Frequencies
# MCEFT : Maximum Coherency Eigenvalue over Frequencies Test


def compute_MCEF_from_C_hats(C_hats):
    max_by_freq = [np.linalg.eigvalsh(C_hats[i])[-1] for i in range(C_hats.shape[0])]
    return np.max(max_by_freq)




############# "Shortcut functions" #############

def compute_MCEF_from_C_hats_sample(C_hats_sample):
    MCEFs = []
    for C_hats in C_hats_sample:
        MCEF = compute_MCEF_from_C_hats(C_hats=C_hats)
        MCEFs.append(MCEF)
    return MCEFs
