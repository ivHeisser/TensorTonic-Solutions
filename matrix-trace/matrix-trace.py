import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    trace = 0

    for i in range(len(A)):
        trace += A[i][i]

    return trace