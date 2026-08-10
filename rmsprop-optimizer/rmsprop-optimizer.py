import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    # Write code here
    s = np.asarray(s, dtype=float)
    w = np.asarray(w, dtype=float)
    g = np.asarray(g, dtype=float)

    if w.shape != g.shape or w.shape != s.shape:
        raise ValueError("w, g and s must have the same shape")

    s_new = beta * s + (1.0 - beta) * np.square(g)
    ''' 
    usually: w_t = w_{t-1} - \frac{\mathrm{lr}}{\sqrt{s_t}+\epsilon}g_t
    epsilon after root(s_t)
    '''
    w_new = w - lr * g / np.sqrt(s_new + eps)
    
    return w_new, s_new