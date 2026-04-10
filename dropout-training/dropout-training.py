import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x)
    if rng is not None:
        random_numbers = rng.random(x.shape)
    else:
        random_numbers = np.random.random(x.shape)

    
    scale = 1.0 / (1.0 - p)
    dropout_pattern = (random_numbers>=p).astype(x.dtype) * scale

    output = x * dropout_pattern
    return output, dropout_pattern