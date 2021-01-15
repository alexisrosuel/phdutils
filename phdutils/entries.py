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
    return (A_sim,B_sim,C_sim,D_sim)




def generate_Y_state_space(N, A_sim, B_sim, C_sim, D_sim):
    """
    Simulate M-dimensional time series given state space model defined by A,B,C,D.
    """
    M = A_sim.shape[0]

    v = complex_gaussian(mean=np.zeros(M), cov=np.identity(M), size=N)
    x = np.zeros((M,N),dtype=complex)#+1j*np.zeros((M,N))
    y = np.zeros((M,N),dtype=complex)#+1j*np.zeros((M,N))

    if np.array_equal(A_sim, np.zeros((M,M))) and np.array_equal(B_sim, np.identity(M)) and np.array_equal(C_sim,np.identity(M)) and np.array_equal(D_sim, np.zeros((M,M))):
        return v

    #initialisation
    x[:,0] = v[:,0]
    y[:,0] = C_sim@x[:,0] + D_sim@v[:,0]

    for i in range(1,N):
        x[:,i] = A_sim@x[:,i-1] + B_sim@v[:,i]
        y[:,i] = C_sim@x[:,i] + D_sim@v[:,i]

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



########## Spike signal #############

def compute_h_k(M, coef_signal):
    """
    Compute the (finite) filtering sequence for the signal model. The scaling is 1/sqrt(M).
    """
    approx = 2000
    return np.sqrt(coef_signal)*np.sqrt(1/M)/np.power(1.1, np.arange(approx))*(1+1j)



def generate_signal(N, M, coef_signal):
    """
    Generate a M-dimensional complex gaussian signal where
    - each dimension is a repetition of a single scalar signal
    - the scalar signal is given by a filtering with the h_k sequence
    """

    if coef_signal == 0:
        return np.zeros((M,N))

    h_k = compute_h_k(M=M, coef_signal=coef_signal)
    approx = h_k.shape[0] # size of the filtering sequence

    epsilon = complex_gaussian(mean=np.zeros(1), cov=np.identity(1), size=N+approx)[0,:]

    real = np.convolve(np.real(h_k), np.real(epsilon), mode='valid') - np.convolve(np.imag(h_k), np.imag(epsilon), mode='valid')
    imag = np.convolve(np.real(h_k), np.imag(epsilon), mode='valid') + np.convolve(np.imag(h_k), np.real(epsilon), mode='valid')

    u = real + 1j*imag
    u = u[:N]
    u = np.repeat(u[np.newaxis,:], M, axis=0)

    return u



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
