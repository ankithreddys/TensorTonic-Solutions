import numpy as np
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        low = bins[i]
        high = bins[i + 1]

        if i < n_bins - 1:
            mask = (y_pred >= low) & (y_pred < high)
        else:
            mask = (y_pred >= low) & (y_pred <= high)
        if np.sum(mask) == 0:
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_pred[mask])
        weight = np.sum(mask) / n
        ece += weight * np.abs(acc - conf)
        
    return ece