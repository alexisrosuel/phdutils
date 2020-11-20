import unittest

import numpy as np

from phdutils.entries import *

class TestEntries(unittest.TestCase):

    def setUp(self):
        pass


    def test_generate_parameters(self):
        M = 2
        theta, beta, delta, gamma = 1, 0.5, 0.2, 0.1

        A_sim,B_sim,C_sim,D_sim = generate_parameters(M, theta, beta, delta, gamma)

        np.array_equal(A_sim, np.array([[theta,0],[beta,theta]]))
        np.array_equal(B_sim, np.array([[1,0],[0,1]]))
        np.array_equal(C_sim, np.array([[1,0],[0,1]]))
        np.array_equal(D_sim, np.array([[gamma,1],[0,gamma]]))


    def test_complex_gaussian(self):
        assert complex_gaussian(np.array([0j]),np.array([[3]]), 100).shape == (1,100)

        Z = complex_gaussian(np.array([0j]),np.array([[3]]), 10000)
        assert np.abs(np.real(Z @ np.conj(Z.T)) / 10000 - 3) < 0.15
        assert np.abs(np.real(Z @ Z.T) / 10000) < 0.15


    # def test_gen_data(self):
    #     assert gen_data(N=100,M=3).shape == (3,100)
    #
    #     autocov = 0.5
    #     Y = gen_data(N=100000,M=1,autocov=autocov)
    #     autocov_hat = np.mean(Y[:,:-1] * np.conj(Y[:,1:]))
    #     autocov_theorique = autocov / (1-autocov**2)
    #     assert np.abs(autocov_hat - autocov_theorique) < 0.02


    def test_generate_Y_state_space(self):
        M, N = 4,2
        nb_repeat = 2
        A_sim, B_sim, C_sim, D_sim = generate_parameters(M=M, theta=0.5, beta=0, delta=0, gamma=0)
        Y = generate_Y_state_space(N=N, A_sim=A_sim, B_sim=B_sim, C_sim=C_sim, D_sim=D_sim)

        assert(Y.shape == (M,N))
        assert(Y[0,0].dtype == np.complex128)


    def test_generate_Y_sample(self):
        M, N = 4,2
        nb_repeat = 2
        A_sim, B_sim, C_sim, D_sim = generate_parameters(M=M, theta=0.5, beta=0, delta=0, gamma=0)
        Ys = generate_Y_sample(N=N, A_sim=A_sim, B_sim=B_sim, C_sim=C_sim, D_sim=D_sim, nb_repeat=nb_repeat)

        assert(Ys.shape == (nb_repeat,M,N))
        assert(Ys[0,0,0].dtype == np.complex128)


if __name__ == '__main__':
    unittest.main()
