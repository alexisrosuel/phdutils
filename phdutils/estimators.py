import numpy as np
import scipy.fftpack
from numba import jit

def get_Fn(N):
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



def S_hat_old(B, Y, nus=None):
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

    if nus is None:
        nus = get_Fn(N)

    S_hats = []
    for nu in nus:
        indice_0 = N*(nu + 0.5) # transformation des fréquences (0 -> -0.5, 1 -> -0.5+1/N, etc.)
        indices = np.array([indice_0 + b for b in np.arange(-B/2,B/2+1,1)])

        indices = indices.astype(int)
        indices = indices % N

        components = fft_Y[:,indices] @ np.conj(fft_Y[:,indices].T)
        S_hats.append(components / (B+1))

    return S_hats




def compute_S_hats(B, Y, nu=None):
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
    M, N = Y.shape
    fft_Y = scipy.fftpack.fft(Y, axis=1)/np.sqrt(N)
    fft_Y = fft_Y[:,:,np.newaxis]
    fft_Y = np.swapaxes(fft_Y, 1, 0)

    if nu is not None:
        indice_0 = N*(nu + 0.5) # transformation des fréquences (0 -> -0.5, 1 -> -0.5+1/N, etc.)
        indices = np.array([indice_0 + b for b in np.arange(-B/2,B/2+1,1)])

        indices = indices.astype(int)
        indices = indices % N

        fft_Y = fft_Y[indices, :, :]

    #periodogram = [fft_Y[:,i,:] @ np.conj(fft_Y[:,i,:].T) for i in range(N)]
    #periodogram = np.array(periodogram)


    periodogram = np.einsum('ijk,imk->ijm', fft_Y, np.conj(fft_Y), optimize=True)
    # fft_Y*np.conj(fft_Y)[:,None,:,0]

    if nu is None:
        h = np.zeros((N,M,M))
        debut, fin = int(N/2-B/2), int(N/2+B/2)+1
        h[debut:fin,:,:] = 1/(fin-debut)

        S_hats = scipy.fftpack.ifft(scipy.fftpack.fft(h, axis=0) * scipy.fftpack.fft(periodogram, axis=0), axis=0)

    else:
        #S_hats = np.mean(periodogram[indices,:,:], axis=0)
        S_hats = np.mean(periodogram, axis=0)
        S_hats = np.array([S_hats])
        #components = fft_Y[:,indices] @ np.conj(fft_Y[:,indices].T)
        #S_hats.append(components / (B+1))

    return S_hats




def compute_C_hats(B, Y, nu=None):
    """
    Calcule l'estimateur de la matrice de cohérence spectrale.

    input:
        nu : fréquence, float (idéalement entre -0.5 et 0.5)
        B : paramètre de lissage du périodogramme
        Y : série temporelle de taille MxN

    output:
        estimateur, matrice de taille MxM
    """
    S_hats = compute_S_hats(B=B, Y=Y, nu=nu)

    C_hats = np.zeros(S_hats.shape)
    for i in range(S_hats.shape[0]):
        diag = np.diagonal(S_hats[i,:,:])
        diag = diag**(-1/2)
        diag = np.diag(diag)
        C_hats[i,:,:] = diag @ S_hats[i,:,:] @ diag
    return C_hats
