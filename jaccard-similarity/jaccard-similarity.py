import numpy as np
def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a = np.asarray(set_a)
    set_b = np.asarray(set_b)

    intersect = len(np.intersect1d(set_a, set_b))
    union = len(np.union1d(set_a,set_b))

    if union != 0:
        return intersect/union
    else:
        return 0