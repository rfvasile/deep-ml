import numpy as np

def simple_conv2d(
    input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int
):

    IH, IW = input_matrix.shape
    KH, KW = kernel.shape

    # Calculate output dimensions
    OH = int((IH + 2 * padding - KH) / stride + 1)
    OW = int((IW + 2 * padding - KW) / stride + 1)

    out = np.zeros(OH * OW).reshape((OH, OW))

    # Apply padding if needed
    padded_input = input_matrix
    if padding:
        PH, PW = IH + 2 * padding, IW + 2 * padding
        padded_input = np.zeros(PH * PW).reshape((PH, PW))

        for h in range(IH):
            for w in range(IW):
                padded_input[h + padding, w + padding] = input_matrix[h, w]

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            sum = 0

            # The convolution op.
            for kph in range(KH):
                for kpw in range(KW):
                    sum += (
                        padded_input[i * stride + kph, j * stride + kpw]
                        * kernel[kph, kpw]
                    )

            out[i][j] = sum
    return out
