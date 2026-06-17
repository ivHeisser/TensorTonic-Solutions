import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.asarray( v, dtype=float )
    norm2 = np.sum( v**2, axis=1 if v.ndim > 1 else 0)
    return np.sqrt(norm2)