import numpy as np


def group_normalization(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5
) -> np.ndarray:
    B, C, H, W = X.shape
    assert C % num_groups == 0

    group_size = int(C / num_groups)
    groups = X.reshape((B, num_groups, group_size, H, W))
    mean = groups.mean(axis=(2, 3, 4), keepdims=True)
    var = groups.var(axis=(2, 3, 4), keepdims=True)
    norm = (groups - mean) / np.sqrt(var + epsilon)
    assert norm.shape == ((B, num_groups, group_size, H, W)), norm.shape

    norm = norm.reshape((B, C, H, W))
    out = gamma * norm + beta
    assert out.shape == (B, C, H, W)

    return out
