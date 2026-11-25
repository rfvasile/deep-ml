import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent of the input array. This function is stable for any input value.
    """
    # We use the absolute value inside the exponential to ensure stability
    # The result is multiplied by the sign, since tanh is an odd function (i.e., tanh(-x) = -tanh(x))
    sign = np.sign(x)

    return sign * (1 - np.exp(-2 * np.abs(x))) / (1 + np.exp(-2 * np.abs(x)))


def dynamic_tanh(x: np.ndarray, alpha: float, gamma: np.ndarray, beta: np.ndarray) -> list[float]:
    """
    Transformers without batch normalization. See https://arxiv.org/pdf/2503.10622
    """
    return np.round((gamma * tanh(alpha * x) + beta), 4).tolist()
