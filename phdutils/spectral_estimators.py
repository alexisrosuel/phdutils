import numpy as np
import scipy.fftpack
from numba import njit, prange





def get_Fn(N):
    """
    Return Fourier frequencies
    """
    return np.arange(-0.5,0.5,1/N)




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
    fft_Y = scipy.fftpack.fft(Y, axis=1)/np.sqrt(N)

    if nus is None:
        nus = get_Fn(N)

    S_hats = []
    for nu in nus:
        indice_0 = N*nu # transformation des fréquences (0 -> -0.5, 1 -> -0.5+1/N, etc.)
        indices = np.array([indice_0 + b for b in np.arange(-B/2,B/2+1,1)])

        indices = indices.astype(int)
        indices = indices % N

        components = fft_Y[:,indices] @ np.conj(fft_Y[:,indices].T)
        S_hats.append(components / (B+1))

    return S_hats


def get_indices(nu, N, B):
    indice_0 = N*nu # transformation des fréquences (0 -> -0.5, 1 -> -0.5+1/N, etc.)
    indices = np.array([indice_0 + b for b in np.arange(-B/2,B/2+1,1)])

    indices = indices.astype(int)
    indices = indices % N
    return indices



@njit('c16[:,:](c16[:,:])', fastmath=False, parallel=False)
def compute_S_hat_from_fft_Y(fft_Y):
    M, B = fft_Y.shape
    S_hat = np.zeros((M,M),dtype='c16')
    for i in range(B):
        for m1 in prange(M):
            S_hat[m1,m1] += (fft_Y[m1,i] * np.conj(fft_Y[m1,i]))/2
            for m2 in range(m1):
                S_hat[m1,m2] += fft_Y[m1,i] * np.conj(fft_Y[m2,i])
    S_hat = (S_hat+np.conj(S_hat.T))
    return S_hat/B



def compute_S_hats(B, Y=None, fft_Y=None, nu=None):
    """
    Calcule l'estimateur smoothed periodogram de la densité spectrale de la série temporelle Y à partir de ou bien:
    - la série Y
    - la fft déjà calculée

    input:
        nu : fréquence, float (idéalement entre -0.5 et 0.5)
        B : paramètre de lissage du périodogramme
        Y : série temporelle de taille MxN

    output:
        estimateur, matrice de taille MxM
    """
    if Y is not None:
        M, N = Y.shape
        fft_Y = scipy.fftpack.fft(Y, axis=1)/np.sqrt(N)

    else:
        M, N = fft_Y.shape

    # ne pas envoyer les indices et tout fft !!! juste le fft sur les indices ça fera moins de calcul + moins de mémoire à deplacer
    if nu is not None:
        indices = get_indices(nu=nu, N=N, B=B)
        S_hat = compute_S_hat_from_fft_Y(fft_Y=fft_Y[:,indices])
        return S_hat[np.newaxis,:,:]

    #periodogram = [fft_Y[:,i,:] @ np.conj(fft_Y[:,i,:].T) for i in range(N)]
    #periodogram = np.array(periodogram)


    if nu is None:
        fft_Y = fft_Y[:,:,np.newaxis]
        fft_Y = np.swapaxes(fft_Y, 1, 0)
        fft_Y = fft_Y[indices, :, :]

        periodogram = np.einsum('ijk,imk->ijm', fft_Y, np.conj(fft_Y), optimize=True)

        h = np.zeros((N,M,M))
        debut, fin = int(N/2-B/2)-1, int(N/2+B/2)+1
        h[debut:fin,:,:] = 1/(fin-debut)

        S_hats = scipy.fftpack.ifft(scipy.fftpack.fft(h, axis=0) * scipy.fftpack.fft(periodogram, axis=0), axis=0)

    #else:
    #    S_hats = np.mean(periodogram, axis=0)
    #    S_hats = np.array([S_hats])

    return S_hats


# def compute_S_hats(B, Y, nu=None):
#     """
#     Calcule l'estimateur de la densité spectrale de la série temporelle Y par la méthode du smoothed periodogram
#     pour toutes les fréquences de Fourier
#
#     input:
#         nu : fréquence, float (idéalement entre -0.5 et 0.5)
#         B : paramètre de lissage du périodogramme
#         Y : série temporelle de taille MxN
#
#     output:
#         estimateur, matrice de taille MxM
#     """
#     M, N = Y.shape
#     fft_Y = scipy.fftpack.fft(Y, axis=1)/np.sqrt(N)
#     fft_Y = fft_Y[:,:,np.newaxis]
#     fft_Y = np.swapaxes(fft_Y, 1, 0)
#
#     if nu is not None:
#         indice_0 = N*nu # transformation des fréquences (0 -> -0.5, 1 -> -0.5+1/N, etc.)
#         indices = np.array([indice_0 + b for b in np.arange(-B/2,B/2+1,1)])
#
#         indices = indices.astype(int)
#         indices = indices % N
#
#         fft_Y = fft_Y[indices, :, :]
#
#     #periodogram = [fft_Y[:,i,:] @ np.conj(fft_Y[:,i,:].T) for i in range(N)]
#     #periodogram = np.array(periodogram)
#     periodogram = np.einsum('ijk,imk->ijm', fft_Y, np.conj(fft_Y), optimize=True)
#
#     if nu is None:
#         h = np.zeros((N,M,M))
#         debut, fin = int(N/2-B/2)-1, int(N/2+B/2)+1
#         h[debut:fin,:,:] = 1/(fin-debut)
#
#         S_hats = scipy.fftpack.ifft(scipy.fftpack.fft(h, axis=0) * scipy.fftpack.fft(periodogram, axis=0), axis=0)
#
#     else:
#         S_hats = np.mean(periodogram, axis=0)
#         S_hats = np.array([S_hats])
#
#     return S_hats




def compute_C_hats(B, fft_Y=None, Y=None, nu=None):
    """
    Calcule l'estimateur de la matrice de cohérence spectrale.

    input:
        nu : fréquence, float (idéalement entre -0.5 et 0.5)
        B : paramètre de lissage du périodogramme
        Y : série temporelle de taille MxN

    output:
        estimateur, matrice de taille MxM
    """
    S_hats = compute_S_hats(B=B, Y=Y, nu=nu, fft_Y=fft_Y)

    C_hats = np.zeros(S_hats.shape)+1j*np.zeros(S_hats.shape)
    for i in range(S_hats.shape[0]):
        diag = np.diagonal(S_hats[i,:,:])
        diag = diag**(-1/2)
        diag = np.diag(diag)
        C_hats[i,:,:] = diag @ S_hats[i,:,:] @ diag
    return C_hats
