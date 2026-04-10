import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    def distance(a,b):
        return np.sqrt(sum((a-b)**2))
    X_train = np.array(X_train)
    X_test  = np.array(X_test)
    
    if X_train.ndim == 1: X_train = X_train.reshape(-1, 1)
    if X_test.ndim  == 1: X_test  = X_test.reshape(-1, 1)
    
    if X_test.size == 0:
        return np.empty((0, k), dtype=int)
        
    result = []
    for test_point in X_test:
        all_distances = [distance(i, test_point) for i in X_train]
        indices = np.argsort(all_distances)[:k]
        padded = np.full(k, -1)
        padded[:len(indices)] = indices
        result.append(padded)
    return np.array(result)