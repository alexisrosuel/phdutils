import numpy as np


def get_Fn(N):
    # définir ce paramètre de manière globale !
    #N = np.min([10,N])
    return np.arange(-0.5,0.5,1/N)



def ksi(Y):
    """
    Calcule la transformée de Fourier renormalisée de la série temporelle Y en les fréquences de Fourier

    input:
        Y : série temporelle de taille MxN
        nu : fréquence, float (idéalement entre -0.5 et 0.5)

    output:
        transformée de Fourier renormalisée de Y en la fréquence réduite nu, de taille Mx1
    """
    fourier_transform = np.fft.fft(Y, norm='ortho') # normalisation en sqrt(N)
    fourier_transform = np.fft.fftshift(fourier_transform) # ramène la fréquence 0 au milieu (-0.5, -0.5+1/N, etc, 0.5)
    return fourier_transform



def S_hat(B, Y, nu=None):
    """
    Calcule l'estimateur de la densité spectrale de la série temporelle Y par la méthode du smoothed periodogram
    pour toutes les fréquences de Fourier

    input:
        nu : fréquence, float (idéalement entre -0.5 et 0.5)
        B : paramètre de lissage du périodogramme
        Y : série temporelle de taille MxN

    output:
        estimateur, matrice de taille MxM
    """
    N = Y.shape[1]
    fft_Y = ksi(Y)

    if nu is None:
        nus = get_Fn(N)
    else:
        nus = [nu]

    S_hats = []
    for nu in nus:
        indice_0 = N*(nu + 0.5) # transformation des fréquences (0 -> -0.5, 1 -> -0.5+1/N, etc.)
        indices = np.array([indice_0 + b for b in np.arange(-B/2,B/2+1,1)])

        indices = indices.astype(int)
        indices = indices % N

        components = fft_Y[:,indices] @ np.conj(fft_Y[:,indices].T)
        S_hats.append(components / (B+1))

    return S_hats



def S_hat_cor(B, Y, nu=None):
    """
    Calcule l'estimateur de la matrice de cohérence spectrale.

    input:
        nu : fréquence, float (idéalement entre -0.5 et 0.5)
        B : paramètre de lissage du périodogramme
        Y : série temporelle de taille MxN

    output:
        estimateur, matrice de taille MxM
    """
    S_hats_computed = S_hat(nu=nu, B=B, Y=Y)

    S_hats_cor = []
    for S_hat_computed in S_hats_computed:
        diag = np.diagonal(S_hat_computed)
        diag = diag**(-1/2)
        diag = np.diag(diag)
        S_hats_cor.append(diag @ S_hat_computed @ diag)
    return S_hats_cor
