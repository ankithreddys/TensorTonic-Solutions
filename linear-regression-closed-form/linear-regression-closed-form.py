import numpy as np

def linear_regression_closed_form(X, y):
    """
    Compute the optimal weight vector using the normal equation.
    """
    # Write code here
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    print(X)
    print(y)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    print(X)
    print(y)

    w = np.linalg.solve(X.T @ X, X.T @ y)
    return w.flatten()