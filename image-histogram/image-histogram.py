import numpy as np
def image_histogram(image):
    """
    Compute the intensity histogram of a grayscale image.
    """
    # Write code here
    image = np.asarray(image)
    bins = np.zeros((256,))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pix_val = image[i][j]
            bins[pix_val] += 1
    return bins.tolist()