import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if num_classes is None:
        num_classes = max(y_true.max(), y_pred.max()) + 1

    cm = np.zeros((num_classes, num_classes), dtype=float)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    if normalize == 'true':
        cm = cm / cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':
        cm = cm / cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':
        cm = cm / cm.sum()
    elif normalize == 'none':
        pass
    else:
        raise ValueError("normalize must be one of: 'none', 'true', 'pred', 'all'")

    return cm

    