import numpy as np
from collections import Counter
import math
def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    y = np.asarray(y)

    if len(np.unique(y)) == 1:
        return 0.0

    value_map = Counter(y)
    total = len(y)

    entropy = 0.0
    for count in value_map.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log(p,2)

    return entropy

        