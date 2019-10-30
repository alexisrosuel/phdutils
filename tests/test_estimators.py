import unittest

import numpy as np

from phdutils.entries import complex_gaussian, build_time_serie
from phdutils.estimators import S_hat, ksi, S_hat_cor, get_Fn

class TestEstimators(unittest.TestCase):
    def setUp(self):
        pass


    # def test_ksi(self):
    #     N=1000
    #     M=3
    #     Y = gen_data(N=N,M=M)
    #     assert ksi(Y).shape == (M,N)


    def test_S_hat(self):
        N=1000
        M=2
        B=100
        burn = 100

        # generate a test MA(1) and a test AR(1)
        eps = complex_gaussian(mean=np.zeros(2), cov=np.identity(2), size=N+burn)[:,burn:]
        test_series = build_time_serie(eps=eps, MA=np.array([0.9,0]), AR=np.array([0,0.9]))

        S_MA = np.absolute(1+0.9*np.exp(2*1j*np.pi*get_Fn(N)))**2 # true S_MA
        S_AR = np.absolute(1/(1-0.9*np.exp(2*1j*np.pi*get_Fn(N))))**2 # true S_AR
        true_S = [np.diag([S_MA[i], S_AR[i]]) for i in range(N)]

        print(true_S[0])

        # true S(\nu)
        S_hat_computed = S_hat(B=B,Y=test_series)
        print(S_hat_computed[int(N/2)])
        error = np.mean([(S_hat_computed[i]-true_S[i])**2 for i in range(N)])

        print([np.real(S_hat_computed[i][1,0]) for i in range(N)])

        print(error)

        assert error<0.01




    #def test_S_hat_cor(self):
        # N=1000
        # M=2
        # Y = gen_data(N=N,M=M,autocov=0.3)
        # s_hat_cor = S_hat_cor(B=100, Y=Y, nu=0.5)
        #
        # assert np.abs(np.real(s_hat_cor[0][0,0])-1) <= 0.0000001
        # assert np.abs(np.real(s_hat_cor[0][1,1])-1) <= 0.0000001
        # assert np.abs(s_hat_cor[0][1,0]) <= 0.3
        #
        #
        # s_hats = []
        # for i in range(1000):
        #     N=1000
        #     M=2
        #     Y = gen_data(N=N,M=M,autocov=0.3, H1=False)
        #     s_hats.append(S_hat_cor(nu=0.2, B=100, Y=Y)[0])
        #
        # s_hat_cor = np.mean(np.array(s_hats), axis=0) # proche d'une matrice diagonale
        # assert np.abs(s_hat_cor[1,0]) <= 0.01




if __name__ == '__main__':
    unittest.main()
