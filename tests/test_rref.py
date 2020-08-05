import unittest
import random as rd
import numpy as np
import sympy as sp

from src import rref


class ReducedRowEchelonFormTest(unittest.TestCase):

    def test_rref(self):
        for _ in range(1024):
            A = np.random.randn(rd.randrange(1, 8), rd.randrange(1, 8))
            expected = np.array(sp.Matrix(A).rref()[0].tolist(), dtype=float)
            actual, _, _ = rref.rref(A)
            self.assertTrue(np.allclose(expected, actual))


if __name__ == '__main__':
    unittest.main()
