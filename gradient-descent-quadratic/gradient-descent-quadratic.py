def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x = x0
    for _ in range(steps):
        d_x = 2 * a * x + b
        x -= lr * d_x

    return x

        