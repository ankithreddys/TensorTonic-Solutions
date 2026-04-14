import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    image = np.asarray(image)
    hist = np.bincount(image.flatten(), minlength=256)
    return hist.tolist()