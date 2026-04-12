import numpy as np

def clip_gradients(g, max_norm):
    """
    Clip gradients using global norm clipping.
    """
    g = np.asarray(g)
    grad_norm = np.sqrt(np.sum(g ** 2))
    print(grad_norm)
    if grad_norm <= max_norm or max_norm <= 0:
        return g
    else:
        return g * (max_norm/grad_norm)