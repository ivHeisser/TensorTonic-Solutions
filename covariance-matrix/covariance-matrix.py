import numpy as np

def covariance_matrix(X):
    """
    Compute sample covariance matrix from dataset X.
    Returns None for invalid input or if N < 2.
    """
    X = np.asarray(X, dtype=float)

    # Check that X is a 2D array
    if X.ndim != 2:
        return None

    N, _ = X.shape

    # Need at least 2 samples for sample covariance
    if N < 2:
        return None

    # Center the data
    mu = np.mean(X, axis=0)
    X_centered = X - mu

    # Sample covariance matrix
    return (X_centered.T @ X_centered) / (N - 1)