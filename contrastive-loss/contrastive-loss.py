import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    # Write code here
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # bring everything to a batch format:
    # (D,) -> (1, D)
    # (N, D) -> (N, D)
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    
    # broadcast a and b to common shape
    a, b = np.broadcast_arrays(a, b)
    
    # y: scalar -> (1,), (N,) stays (N,)
    y = np.atleast_1d(y)
    
    if y.shape[0] != a.shape[0]:
        raise ValueError(
            f"y must have length {a.shape[0]}, got {y.shape[0]}"
        )
    
    d = np.linalg.norm(a - b, axis=1)
    loss = y * d**2 + (1-y) * np.maximum(0, margin-d)**2
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss