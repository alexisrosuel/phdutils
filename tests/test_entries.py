import unittest

import numpy as np

from phdutils.entries import complex_gaussian, gen_data

class TestEntries(unittest.TestCase):

    def setUp(self):
        pass

    def test_complex_gaussian(self):
        assert complex_gaussian(np.array([0j]),np.array([[3]]), 100).shape == (1,100)

        Z = complex_gaussian(np.array([0j]),np.array([[3]]), 10000)
        assert np.abs(np.real(Z @ np.conj(Z.T)) / 10000 - 3) < 0.05
        assert np.abs(np.real(Z @ Z.T) / 10000) < 0.05


    # def test_gen_data(self):
    #     assert gen_data(N=100,M=3).shape == (3,100)
    #
    #     autocov = 0.5
    #     Y = gen_data(N=100000,M=1,autocov=autocov)
    #     autocov_hat = np.mean(Y[:,:-1] * np.conj(Y[:,1:]))
    #     autocov_theorique = autocov / (1-autocov**2)
    #     assert np.abs(autocov_hat - autocov_theorique) < 0.02


if __name__ == '__main__':
    unittest.main()
