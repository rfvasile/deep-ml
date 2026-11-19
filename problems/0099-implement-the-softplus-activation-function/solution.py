import numpy as np


def softplus(x: float) -> float:
    """
    Compute the softplus activation function.
    """
    val = np.log(1 + np.exp(x))
    return round(val, 4)
