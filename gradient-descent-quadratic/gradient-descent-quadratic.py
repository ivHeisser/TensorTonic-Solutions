def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    x = x0
    div_f = lambda x: 2.0 * a * x + b
    
    for i in range(steps):
        x -= lr * div_f(x)
    
    return x