import numpy as np


def relu(x):
    return np.where(x > 0, x, 0)


def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
    """Residual block with shortcut connection"""
    y1 = x @ w1.T
    z1 = relu(y1)

    y2 = z1 @ w2.T
    z2 = relu(y2)

    y3 = z2 + x
    z3 = relu(y3)
    return z3
