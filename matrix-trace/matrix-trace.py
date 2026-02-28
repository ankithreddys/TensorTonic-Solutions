import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    A = np.asarray(A)
    n = A.shape[0]
    
    trace_sum = 0.0
    for i in range(n):
        trace_sum += A[i, i]
        
    return trace_sum