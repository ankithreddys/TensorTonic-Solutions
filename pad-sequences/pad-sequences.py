import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    max_len = max_len if max_len else len(max(seqs, key=len))
    final_seq = []
    for n_seq in seqs:
        if len(n_seq) <= max_len:
            full_arr = np.append(np.asarray(n_seq), np.full(max_len - len(n_seq), pad_value))
        else:
            extra = len(n_seq) - max_len
            full_arr = np.asarray(n_seq)[:-extra]
        final_seq.append(full_arr.tolist())
    return final_seq