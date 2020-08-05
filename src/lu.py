import numpy as np


def forward_substitution(A):
    """Transform a matrix into a upper triangular or trapezoidal matrix.

    Args:
        A: A matrix.

    Returns:
        L: A lower triangular or trapezoidal matrix with unit diagonal.
        U: An upper triangular or trapezoidal matrix.
    """
    m, n = A.shape
    i = j = 0
    L, U = np.eye(m), np.copy(A)

    while i < m and j < n:
        # Find the first row with nonzero pivot.
        k = i
        while k < m and U[k, j] == 0:
            k += 1

        if k < m:
            # Swap the current row with the first row with nonzero pivot.
            U[i], U[k] = U[k], U[i]
            # Now, we use the row containing the pivot to eliminate all
            # other values underneath in the same column.
            for x in range(i+1, m):
                U[x] = U[x] - U[x, j]*(U[i]/U[i, j])
            i += 1

        j += 1

    return L, U
