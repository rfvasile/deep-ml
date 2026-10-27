import numpy as np

def softmax(scores: list[float]) -> list[float]:
    """Compute the stable softmax activation function."""
    if not scores:
        return []

    scores = np.array(scores)
    scores_max = scores.max()

    exp_scores = np.exp(scores - scores_max)
    out = exp_scores / exp_scores.sum()

    return out.round(4).tolist()
