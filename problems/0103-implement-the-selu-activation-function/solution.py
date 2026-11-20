import numpy as np


def selu(x: float) -> float:
    """
    Implements the SELU (Scaled Exponential Linear Unit) activation function.
    """
    alpha = 1.6732632423543772
    scale = 1.0507009873554804

    rhs = np.where(x > 0, x, alpha * (np.exp(x) - 1))

    return scale * rhs
