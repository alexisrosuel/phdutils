import unittest

import numpy as np

from phdutils.entries import generate_Y_state_space, generate_parameters
from phdutils.estimators import *

class TestEstimators(unittest.TestCase):
    def setUp(self):
        pass


    # def test_ksi(self):
    #     N=1000
    #     M=3
    #     Y = gen_data(N=N,M=M)
    #     assert ksi(Y).shape == (M,N)


    def test_compute_C_hats(self):
        N, M, B = 1000, 2, 100
        burn = 100

        # generate independent time series
        A_sim, B_sim, C_sim, D_sim = generate_parameters(M=M, theta=0.5, beta=0, delta=0, gamma=0)
        Y = generate_Y_state_space(N=N, A_sim=A_sim, B_sim=B_sim, C_sim=C_sim, D_sim=D_sim)

        C_hats = compute_C_hats(Y=Y, B=B)

        assert(C_hats.shape == (N,M,M))

        ecart_quadratique = [np.sum(np.abs(C_hat - np.identity(2))**2) for C_hat in C_hats]
        assert(np.max(ecart_quadratique) < 0.2)



    # def test_compute_S_hats(self):
    #     N=1000
    #     M=2
    #     B=100
    #     burn = 100
    #
    #     # generate a test MA(1) and a test AR(1)
    #     eps = generate_Y_state_space(mean=np.zeros(2), cov=np.identity(2), size=N+burn)[:,burn:]
    #     test_series = build_time_serie(eps=eps, MA=np.array([0.9,0]), AR=np.array([0,0.9]))
    #
    #     S_MA = np.absolute(1+0.9*np.exp(2*1j*np.pi*get_Fn(N)))**2 # true S_MA
    #     S_AR = np.absolute(1/(1-0.9*np.exp(2*1j*np.pi*get_Fn(N))))**2 # true S_AR
    #     true_S = [np.diag([S_MA[i], S_AR[i]]) for i in range(N)]
    #
    #     print(true_S[0])
    #
    #     # true S(\nu)
    #     S_hat_computed = S_hat(B=B,Y=test_series)
    #     print(S_hat_computed[int(N/2)])
    #     error = np.mean([(S_hat_computed[i]-true_S[i])**2 for i in range(N)])
    #
    #     print([np.real(S_hat_computed[i][1,0]) for i in range(N)])
    #
    #     print(error)
    #
    #     assert error<0.01







if __name__ == '__main__':
    unittest.main()
