import numpy as np
import scipy.linalg
import scipy.stats



def complex_gaussian(mean, cov, size):
    """
    Generate iid cicurlary complex gaussian sample

    input:
        mean:
        cov:
        size:

    return:
        (M,N) matrix

    """
    mean_split = np.concatenate([np.real(mean), np.imag(mean)])
    cov_split = 1/2 * np.concatenate([np.concatenate([np.real(cov), -np.imag(cov)], axis=0),
                                      np.concatenate([np.imag(cov),  np.real(cov)], axis=0)], axis=1)

    XY = np.random.multivariate_normal(mean=mean_split, cov=cov_split, size=size)
    X = XY[:,:mean.shape[0]]
    Y = XY[:,mean.shape[0]:]

    # transpose the generated data to match the paper notations: the M-dimensional individuals are column-stacked (size MxN)
    return (X + 1j * Y).T




def build_time_serie(eps, MA, AR):
    y = (0+0*1j)*np.zeros(eps.shape)
    y[:,0] = eps[:,0]

    for i in range(1,y.shape[1]):
        y[:,i] = AR@y[:,i-1] + MA@eps[:,i-1] + eps[:,i]

    return y



def gen_data(N, M, cov=None, MA=None, AR=None, burn=100):
    """
    input:
        N : nombres d'échantillon
        M : dimension de la série temp
        cov: noise covariance
        MA: matrix valued MA(1) coef
        AR: matrix valued AR(1) coef
        burn: samples to reject

    return:
        matrice Y de taille MxN (1 ligne = 1 individu suivi au cours du temps)
    """
    if cov is None:
        cov = np.identity(M)
    if MA is None:
        MA = np.zeros(M)
    if AR is None:
        AR = np.zeros(M)

    eps = complex_gaussian(mean=np.zeros(M), cov=cov, size=N+burn)

    Y = build_time_serie(eps=eps, MA=MA, AR=AR)
    return Y[:,burn:]



def generate_y_state_space(N, M, A, B, C, D):
    v = complex_gaussian(mean=np.zeros(M), cov=np.identity(M), size=N)
    x = np.zeros((M,N),dtype=complex)#+1j*np.zeros((M,N))
    y = np.zeros((M,N),dtype=complex)#+1j*np.zeros((M,N))

    #initialisation
    x[:,0] = v[:,0]
    y[:,0] = C@x[:,0] + D@v[:,0]

    for i in range(1,N):
        x[:,i] = A@x[:,i-1] + B@v[:,i]
        y[:,i] = C@x[:,i] + D@v[:,i]

    return y
