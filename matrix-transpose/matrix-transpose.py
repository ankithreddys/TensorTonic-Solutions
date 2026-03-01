import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A = np.asarray(A)
    N, D = A.shape

    T = np.zeros((D,N))

    for i in range(N):
        for j in range(D):
            T[j,i] = A[i,j]

    return T
