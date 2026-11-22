# %%
import numpy as np


def batch_normalization(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray | None = None,
    running_var: np.ndarray | None = None,
    momentum: float = 0.1,
    epsilon: float = 1e-5,
    training: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform Batch Normalization on BCHW input.

    Args:
        X: Input array of shape (B, C, H, W)
        gamma: Scale parameter of shape (1, C, 1, 1)
        beta: Shift parameter of shape (1, C, 1, 1)
        running_mean: Running mean for inference, shape (1, C, 1, 1)
        running_var: Running variance for inference, shape (1, C, 1, 1)
        momentum: Momentum for updating running statistics (following PyTorch convention)
        epsilon: Small constant for numerical stability
        training: If True, use batch statistics; if False, use running statistics

    Returns:
        Tuple of (normalized_output, updated_running_mean, updated_running_var)
    """
    B, C, H, W = X.shape
    if training:
        mean = 1 / (B * H * W) * X.sum(axis=(0, 2, 3), keepdims=True)
        variance = 1 / (B * H * W) * ((X - mean) ** 2).sum(axis=(0, 2, 3), keepdims=True)
        normalized = (X - mean) / (np.sqrt(variance + epsilon))
        y = gamma * normalized + beta

        rm = 0 if running_mean is None else running_mean
        rv = 1 if running_var is None else running_var
        
        running_mean = (1 - momentum) * rm + momentum * mean
        running_var = (1 - momentum) * rv + momentum * variance
    else:
        assert running_var is not None
        normalized = (X - running_mean) / (np.sqrt(running_var + epsilon))
        y = gamma * normalized + beta

    assert running_mean is not None
    assert running_var is not None
    return y, running_mean, running_var
