import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)  
    N, D = X.shape

    # target parameters:
    w = np.zeros(D)
    b = 0.0
    
    # loss function parameters:
    losses = []
    tol = 1e-6
    eps = 1e-15  # log(0) protection

    for _ in range(steps):
        # Forward pass
        p = _sigmoid(X@w + b) # p=prediction

        # Loss
        p_clipped = np.clip(p, eps, 1-eps)
        loss = -np.mean(
            y * np.log(p_clipped) +
            (1 - y) * np.log(1 - p_clipped)
        )
        losses.append(loss)
        if len(losses) > 1 and abs(losses[-2] - losses[-1]) < tol: # early stopping
            '''
            another early stopping conditions:
            if np.linalg.norm(grad_w) < tol:
            if max(np.abs(grad_w)) < tol:
            '''
            break
        
        # Gradients
        grad_w = (X.T@(p - y))/N
        grad_b = np.mean(p - y)

        # Parameter update
        w -= lr*grad_w
        b -= lr*grad_b
    
    return w, b#, losses