import numpy as np


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Compute the hyperbolic tangent of the input array.
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def dynamic_tanh(x: np.ndarray, alpha: float, gamma: np.ndarray, beta: np.ndarray) -> list[float]:
    """
    Transformers without batch normalization. See https://arxiv.org/pdf/2503.10622
    """
    return np.round((gamma * tanh(alpha * x) + beta), 4).tolist()
