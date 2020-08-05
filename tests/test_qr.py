import unittest
import random as rd
import numpy as np

from src import qr


class QRTest(unittest.TestCase):

    def test_gram_schmidt(self):
        for _ in range(1024):
            m, n = rd.randrange(1, 8), rd.randrange(1, 8)
            A = np.random.randn(m, n)
            expected_q, expected_r = np.linalg.qr(A)
            actual_q, actual_r = qr.gram_schmidt(A)
            self.assertTrue(np.allclose(expected_q, actual_q))
            self.assertTrue(np.allclose(expected_r, actual_r))


if __name__ == '__main__':
    unittest.main()
