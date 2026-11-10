import numpy as np


def pos_encoding(position, d_model):
    if position == 0 or d_model <= 0:
        return -1

    A = np.zeros((position, d_model), dtype=np.float32)

    for pos in range(position):
        for i in range(d_model):
            angle = pos / 10000 ** ((2 * (i // 2)) / d_model)
            A[pos, i] = np.sin(angle) if i % 2 == 0 else np.cos(angle)

    return A.astype(np.float16)


def pos_encoding2(position, d_model):
    pos = np.arange(position)[:, None]  # (P, 1)
    d = np.arange(d_model)[None, :]  # (1, D)
    angle = pos / 10000 ** ((2 * (d // 2)) / d_model)
    angle[:, 0::2] = np.sin(angle[:, 0::2])
    angle[:, 1::2] = np.cos(angle[:, 1::2])
    return angle.astype(np.float16)
