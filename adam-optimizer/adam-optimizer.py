import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    param = np.asarray(param, dtype=float)
    grad = np.asarray(grad, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    
    # Step 1: Update First Moment 
    m = beta1 * m + (1 - beta1) * grad
    # Step 2: Update Second Moment
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    # Step 3: Bias Correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    # Step 4: Parameter Update
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v