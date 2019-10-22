import unittest

import numpy as np

from utils.entries import gen_data
from utils.estimators import S, S_hat, ksi, Sigma, S_hat_cor, S_hat_mcor, get_Fn

class TestEstimators(unittest.TestCase):

    def setUp(self):
        pass

    def test_S(self):
        assert S(1, np.zeros(1)) == 1
        assert S(0.5, 0.4*np.ones(1)) == 1/(1+2*0.4+0.4**2)

        M=10
        S_calculé = S(nu=0.5, H=np.identity(M))
        assert S_calculé.shape == (M,M)


    def test_ksi(self):
        N=1000
        M=3
        Y = gen_data(N=N,M=M)
        assert ksi(Y).shape == (M,N)


    def test_S_hat(self):
        N=1000
        M=2
        B=2

        s_hat = (1j*0)*np.zeros((M,M))
        for i in range(1000):
            Y, H = gen_data(N=N,M=M,autocov=0.7, return_infos=True)
            s_hat += S_hat(B=B,Y=Y)[0][0,0]/1000
        s_theorique = S(nu=0.5, H=H)

        assert np.max(s_theorique - s_hat) < 0.01
        # notre estimateur S_hat est correct, en moyenne les coeffs vont tous vers leur espérance

        N=1000
        M=3
        Y = gen_data(N=N,M=M,autocov=0.3)
        assert S_hat(nu=0.2,B=6,Y=Y)[0].shape == (M,M)

        N=1000
        M=3
        Y = gen_data(N=N,M=M,autocov=0.3)
        assert len(S_hat(B=6,Y=Y)) == get_Fn(N).shape[0]


    def test_Sigma(self):
        N=1000
        M=2
        Y = gen_data(N=N,M=M)
        assert Sigma(0.3,Y,B=6).shape == (2,7)


    def test_S_hat_cor(self):
        N=1000
        M=2
        Y = gen_data(N=N,M=M,autocov=0.3)
        s_hat_cor = S_hat_cor(B=100, Y=Y, nu=0.5)

        assert np.abs(np.real(s_hat_cor[0][0,0])-1) <= 0.0000001
        assert np.abs(np.real(s_hat_cor[0][1,1])-1) <= 0.0000001
        assert np.abs(s_hat_cor[0][1,0]) <= 0.3


        s_hats = []
        for i in range(1000):
            N=1000
            M=2
            Y = gen_data(N=N,M=M,autocov=0.3, H1=False)
            s_hats.append(S_hat_cor(nu=0.2, B=100, Y=Y)[0])

        s_hat_cor = np.mean(np.array(s_hats), axis=0) # proche d'une matrice diagonale
        assert np.abs(s_hat_cor[1,0]) <= 0.01

    def test_S_hat_mcor(self):
        N=100000
        M=2
        Y, H = gen_data(N=N,M=M,autocov=0.3, return_infos=True)
        s_hat_mcor = S_hat_mcor(nu=0.2, B=100, Y=Y, H=H)

        # la diagonale vaut presque 1
        assert np.abs(s_hat_mcor[0][0,0]-1) <= 0.3
        assert np.abs(s_hat_mcor[0][1,1]-1) <= 0.3


if __name__ == '__main__':
    unittest.main()
