import numpy as np


def angle_rads(position, d_model):
    A = np.zeros((position, d_model), dtype=np.float32)

    for pos in range(position):
        for i in range(d_model):
            angle = pos / 10000 ** ((2 * (i // 2)) / d_model)
            A[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)

    return A.astype(np.float16)


def pos_encoding(position, d_model):
    if position == 0 or d_model <= 0:
        return -1
    return angle_rads(position, d_model)
