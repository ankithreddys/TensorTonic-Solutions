import numpy as np
def conv2d(image, kernel, stride=1, padding=0):
    """
    Apply 2D convolution to a single-channel image.
    """
    # Write code here
    image = np.asarray(image)
    kernel = np.asarray(kernel)

    image_H, image_W = image.shape
    kernel_H, kernel_W = kernel.shape
    H_out = ((image_H - kernel_H + 2 * padding) // stride) + 1
    W_out = ((image_W - kernel_W + 2 * padding) // stride) + 1

    output = np.zeros((H_out, W_out))

    if padding > 0:
        image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride

            image_part = image[h_start:h_start + kernel_H,
                               w_start:w_start + kernel_W]

            output[i, j] = np.sum(image_part * kernel)

    return output.tolist()

    