import unittest
import random as rd
import numpy as np

from src import lu


class LUTest(unittest.TestCase):

    def test_forward_substitution(self):
        for _ in range(1024):
            m, n = rd.randrange(1, 8), rd.randrange(1, 8)
            A = np.random.randn(m, n)
            expected_l, expected_u = np.linalg.lu(A)
            actual_l, actual_u = lu.forward_substitution(A)
            self.assertTrue(np.allclose(expected_l, actual_l))
            self.assertTrue(np.allclose(expected_u, actual_u))


if __name__ == '__main__':
    unittest.main()
