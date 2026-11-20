import numpy as np


def swish(x: float) -> float:
    """
    Implements the Swish activation function.
    """
    return x / (1 + np.exp(-x))
