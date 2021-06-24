import numpy as np
import scipy.linalg
import scipy.stats
from numba import njit


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
    if np.all(mean == np.zeros(mean.shape[0])) and np.all(cov == np.identity(mean.shape[0])):
        XY = np.random.normal(loc=0,scale=1/np.sqrt(2),size=(size,2*mean.shape[0]))

    else:
        mean_split = np.concatenate((np.real(mean), np.imag(mean)))
        cov_split = 1/2 * np.concatenate((np.concatenate((np.real(cov), -np.imag(cov)), axis=0),
                                          np.concatenate((np.imag(cov),  np.real(cov)), axis=0)), axis=1)

        XY = np.random.multivariate_normal(mean=mean_split, cov=cov_split, size=size)

    X = XY[:,:mean.shape[0]]
    Y = XY[:,mean.shape[0]:]

    # transpose the generated data to match the paper notations: the M-dimensional individuals are column-stacked (size MxN)
    return (X + 1j * Y).T



@njit('UniTuple(c16[:,:], 4)(int64, c16, c16, f8, f8)')
def generate_parameters(M, theta=0, beta=0, delta=0, gamma=0):
    """
    Parameters for the state space model representation of the time series
    """
    A_sim = theta*np.identity(M) + beta*np.diag(np.ones(M-1), k=-1)
    B_sim = np.identity(M)
    C_sim = np.identity(M)
    D_sim = np.zeros((M,M))
    for i in range(M):
        D_sim += (np.diag(gamma**(M-i-1)*np.ones(M-i), k=i))
    D_sim *= delta
    A_sim, B_sim, C_sim, D_sim = A_sim.astype(np.cdouble),B_sim.astype(np.cdouble),C_sim.astype(np.cdouble),D_sim.astype(np.cdouble)
    return (A_sim, B_sim, C_sim, D_sim)




# def generate_Y_state_space(N, A_sim, B_sim, C_sim, D_sim):
#     """
#     Simulate M-dimensional time series given state space model defined by A,B,C,D.
#     """
#     M = A_sim.shape[0]
#
#     v = complex_gaussian(mean=np.zeros(M), cov=np.identity(M), size=N)
#     x = np.zeros((M,N),dtype=complex)#+1j*np.zeros((M,N))
#     y = np.zeros((M,N),dtype=complex)#+1j*np.zeros((M,N))
#
#     if np.array_equal(A_sim, np.zeros((M,M))) and np.array_equal(B_sim, np.identity(M)) and np.array_equal(C_sim,np.identity(M)) and np.array_equal(D_sim, np.zeros((M,M))):
#         return v
#
#     #initialisation
#     x[:,0] = v[:,0]
#     y[:,0] = C_sim@x[:,0] + D_sim@v[:,0]
#
#     for i in range(1,N):
#         x[:,i] = A_sim@x[:,i-1] + B_sim@v[:,i]
#         y[:,i] = C_sim@x[:,i] + D_sim@v[:,i]
#
#     return y



@njit('c16[:,::1](int64, c16[:,::1], c16[:,::1], c16[:,::1], c16[:,::1])', fastmath=False)
def generate_Y_state_space(N, A_sim, B_sim, C_sim, D_sim):
    """
    Simulate M-dimensional time series given state space model defined by A,B,C,D.
    """
    M = A_sim.shape[0]

    v = np.random.normal(loc=0,scale=1/np.sqrt(2),size=(M,2*N))
    v = v[:,:N] + 1j*v[:,N:]

    #v = complex_gaussian(mean=np.zeros(M), cov=np.identity(M), size=N)
    x = np.empty((M,N),dtype='c16')#+1j*np.zeros((M,N))
    y = np.empty((M,N),dtype='c16')#+1j*np.zeros((M,N))

    #if np.array_equal(A_sim, np.zeros((M,M))) and np.array_equal(B_sim, np.identity(M)) and np.array_equal(C_sim,np.identity(M)) and np.array_equal(D_sim, np.zeros((M,M))):
    #    return v

    Dv = D_sim@v
    Bv = B_sim@v

    # state computations
    x[:,0] = v[:,0]
    for i in range(1,N):
        x[:,i] = A_sim@x[:,i-1] + Bv[:,i]

    y = C_sim@x + Dv

    return y



def generate_Y_sample(N, A_sim, B_sim, C_sim, D_sim, nb_repeat):
    '''
    Generate independent nb_repeat of time series with parameter N, M, A_sim, B_sim, C_sim, D_sim
    return Ys[nb_repeat, M, N]
    '''
    Ys = []
    for repeat in range(nb_repeat):
        print('Simulating time series : %s / %s' % (repeat, nb_repeat) , end='\r')
        Y = generate_Y_state_space(N=N, A_sim=A_sim, B_sim=B_sim, C_sim=C_sim, D_sim=D_sim)
        Ys.append(Y)
    return np.array(Ys)






##################### OLD VERSION ####################


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



def gen_y_ar(N,M,theta):
    epsilons = complex_gaussian(mean=np.zeros(M), cov=np.identity(M), size=N)
    y = epsilons
    for i in range(1,N):
        y[:,i] = theta*y[:,i-1] + epsilons[:,i]
    return y
