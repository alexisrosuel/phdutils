import unittest

import numpy as np

from phdutils.misc import *

class TestMisc(unittest.TestCase):

    def setUp(self):
        pass

    def test_compute_repartition(self):
        sample_test = np.array([1,2,2,3,4])
        y_range = np.array([1,2,3,4])

        np.array_equal(compute_repartition(sample=sample_test, y_range=y_range), np.array([0,1/5,3/5,4/5]))


if __name__ == '__main__':
    unittest.main()
