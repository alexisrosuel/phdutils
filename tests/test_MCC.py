import unittest

import numpy as np

from phdutils.MCC import *

class TestMCC(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_MCC(self):
        C_hats_1 = np.array([[1,0.5+0.5*1j],[0.5*1j,1]])
        C_hats_2 = np.array([[1,0.1+0.1*1j],[1+1*1j,1]])

        C_hats_test = [C_hats_1, C_hats_2]
        mask_test = [1,0]
        np.array_equal(compute_MCC(C_hats_test, mask_test), np.abs(0.5+0.5*1j)**2)
        mask_test = [0,1]
        np.array_equal(compute_MCC(C_hats_test, mask_test), np.abs(1+1*1j)**2)

    def test_mask(self):
        np.array_equal(create_mask(N=10,B=2), np.array([1., 0., 0., 1., 0., 0., 1., 0., 0., 0.]))


    def test_scale_MCC(self):
        N, M, B = 100, 3, 10
        MCC = np.array([1,2,3])

        scaled_MCC = scale_MCC(N, B, M, MCC)

        assert(scaled_MCC, (B+1)*MCC -np.log(N/(B+1)) -np.log(M*(M-1)/2) )
        


if __name__ == '__main__':
    unittest.main()
