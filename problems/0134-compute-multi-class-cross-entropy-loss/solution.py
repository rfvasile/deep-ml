import numpy as np


def compute_cross_entropy_loss(predicted_probs: np.ndarray, true_labels: np.ndarray, epsilon=1e-15) -> float:
    """Implement categorial cross entropy."""
    assert predicted_probs.ndim == 2, "expected batch and class dimensions"
    assert true_labels.ndim == 2, "expected batch and class dimensions"

    sum = 0
    for i in range(predicted_probs.shape[0]):
        for j in range(predicted_probs.shape[1]):
            term = true_labels[i, j] * np.log(predicted_probs[i, j] + epsilon)
            sum += term

    return np.round(-sum / predicted_probs.shape[0], 4)
