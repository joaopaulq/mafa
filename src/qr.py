import numpy as np


def gram_schmidt(A):
    """Computes the QR decomposition of a matrix using the Gram-Schmidt process.

    Args:
        A: A matrix.

    Returns:
        Q: An orthonormal matrix.
        R: An upper triangular or trapezoidal matrix.
    """
    m, n = A.shape
    dim = min(m, n)
    Q = np.copy(A)
    R = np.zeros((dim, n))

    norm = np.linalg.norm(A[:, 0])
    Q[:, 0] = A[:, 0] / norm
    R[0, 0] = norm

    for j in range(1, dim):
        for k in range(j-1, -1, -1):
            dot = np.dot(A[:, j], Q[:, k])
            Q[:, j] = Q[:, j] - dot*Q[:, k]
            R[k, j] = dot

        norm = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / norm
        R[j, j] = norm

    if n > m:
        for j in range(dim, n):
            R[:, j] = np.linalg.solve(Q[:, :m], Q[:, j])

    return Q[:, :dim], R
