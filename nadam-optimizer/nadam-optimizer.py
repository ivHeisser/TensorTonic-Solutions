import numpy as np

def nadam_step(w, m, v, grad, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Perform one Nadam update step.
    Key idea: use bias-corrected moments
        and “lookahead” for momentum

    Impotant problems:
        1. there is no "t" (iteration).
        Nadam definitely requires a "t" !
        Without it:
            * no bias correction
            * the formula is incomplete
        2. dual use of grad:
            beta1 * m + (1-beta1) * grad
        But "m" already contains a moving average,
        "grad" is added again → it's not Nadam !
    """
    # Write code here
    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # Step 1: Update First Moment
    m = beta1 * m + (1-beta1) * grad
    # Step 2: Update Second Moment
    v = beta2 * v + (1-beta2) * (grad**2)
    # Step 3: Nesterov-Adjusted Update
    w = w - lr * (beta1 * m + (1-beta1) * grad) / (np.sqrt(v) + eps)

    return w, m, v


def nadam_step_corrected(w, m, v, grad, t,
               lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):

    w = np.asarray(w, dtype=float)
    m = np.asarray(m, dtype=float)
    v = np.asarray(v, dtype=float)
    grad = np.asarray(grad, dtype=float)

    # First moment
    m = beta1 * m + (1 - beta1) * grad

    # Second moment
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # Nesterov-accelerated gradient (Nadam)
    m_nesterov = beta1 * m_hat + (1 - beta1) * grad / (1 - beta1 ** t)

    # Update
    w = w - lr * m_nesterov / (np.sqrt(v_hat) + eps)

    return w, m, v