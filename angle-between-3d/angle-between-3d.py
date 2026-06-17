import numpy as np

def angle_between_3d(v, w):
    """
    Compute the angle (in radians) between two 3D vectors.
    """
    # Your code here  
    v = np.asarray(v, dtype=float)
    w = np.asarray(w, dtype=float)

    if v.shape != (3,) or w.shape != (3,):
        raise ValueError("Both vectors must be 3-dimensional")

    norm_v = np.sqrt(np.sum(v**2))
    norm_w = np.sqrt(np.sum(w**2))

    if norm_v <= 1e-10 or norm_w <= 1e-10:
        return np.nan
    
    cos_theta = (v @ w) / (norm_v * norm_w)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)