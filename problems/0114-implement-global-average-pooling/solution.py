import numpy as np


def global_avg_pool(x: np.ndarray) -> np.ndarray:
    """ "Global average poolin function that computes the average of each feature map in the input tensor."""
    return np.mean(x, axis=(0, 1))
