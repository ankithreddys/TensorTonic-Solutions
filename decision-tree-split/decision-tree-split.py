import numpy as np
from collections import Counter

def decision_tree_split(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    def gini(labels):
        counts = Counter(labels)
        n = len(labels)
        return 1 - sum((c / n) ** 2 for c in counts.values())

    n_samples, n_features = X.shape

    # Parent Gini
    gini_parent = gini(y)

    best_gain = -1
    best_feature = None
    best_threshold = None

    for feature_idx in range(n_features):
        values = np.sort(np.unique(X[:, feature_idx]))

        thresholds = (values[:-1] + values[1:]) / 2

        for threshold in thresholds:
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            gini_left = gini(y_left)
            gini_right = gini(y_right)

            gini_split = (len(y_left) / n_samples) * gini_left + \
                         (len(y_right) / n_samples) * gini_right

            gain = gini_parent - gini_split

            if (gain > best_gain or
                (gain == best_gain and feature_idx < best_feature) or
                (gain == best_gain and feature_idx == best_feature and threshold < best_threshold)):
                
                best_gain = gain
                best_feature = feature_idx
                best_threshold = threshold

    return [best_feature, best_threshold]