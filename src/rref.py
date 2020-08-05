import numpy as np


def rref(A):
    """Computes the reduced row echelon form of a matrix.

    Args:
        A: A m by n matrix.

    Returns:
        R: The matrix A in reduced row echelon form.
        pivot_columns: List containing the pivot columns number.
        free_columns: List containing the free columns number.
    """
    U, pivot_columns, free_columns = forward_substitution(A)
    R = _back_substitution(U, pivot_columns)

    return R, pivot_columns, free_columns


def forward_substitution(A):
    """Transform a matrix into a upper triangular or trapezoidal matrix.

    Args:
        A: A m by n matrix.

    Returns:
        U: An upper triangular or trapezoidal matrix.
        pivot_columns: List containing the pivot columns number.
        free_columns: List containing the free columns number.
    """
    m, n = A.shape
    pivot_columns = []
    free_columns = []
    i = j = 0
    U = np.copy(A)

    while i < m and j < n:
        # Find the first row with nonzero pivot.
        k = i
        while k < m and U[k, j] == 0:
            k += 1

        if k < m:
            pivot_columns.append(j)
            # Swap the current row with the first row with nonzero pivot.
            U[i], U[k] = U[k], U[i]
            # Now, we use the row containing the pivot to eliminate all
            # other values underneath in the same column.
            for x in range(i+1, m):
                U[x] = U[x] - U[x, j]*(U[i]/U[i, j])
            i += 1
        else:
            free_columns.append(j)

        j += 1

    # In case that the matrix A has more columns than rows, i.e. m < n.
    while j < n:
        free_columns.append(j)
        j += 1

    return U, pivot_columns, free_columns


def _back_substitution(U, pivot_columns):
    """Produce zeros above the pivots.

    Args:
        U: An upper triangular or trapezoidal matrix.
        pivot_columns: List containing the pivot columns number.

    Returns:
        The matrix U with zeros above the pivots.
    """
    i = len(pivot_columns) - 1
    
    for j in reversed(pivot_columns):
        U[i] = U[i] / U[i, j]
        for k in range(i-1, -1, -1):
            U[k] = U[k] - U[k, j]*U[i]
        i -= 1

    return U
