import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.zeros((seq_len, d_model))

    for pos in range(seq_len):
        for i in range(d_model//2):
            denominator = base ** (2*i / d_model)
            PE[pos, 2 * i] = np.sin(pos/denominator)
            PE[pos, 2 * i + 1] = np.cos(pos/denominator)

        if d_model % 2 != 0:
            i_last = d_model // 2 
            denominator = base ** (2 * i_last / d_model)
            PE[pos, d_model - 1] = np.sin(pos / denominator)

    return PE