import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    if not A:
        return np.array([])

    rows = len(A)
    cols = len(A[0])

    A_T = np.zeros((cols, rows)) 

    for i in range(rows):
        for j in range(cols):
            A_T[j][i] = A[i][j]

    return A_T
