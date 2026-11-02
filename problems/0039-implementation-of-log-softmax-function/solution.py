import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    x = np.asarray(scores)
    shifted = x - x.max()
    return shifted - np.log(np.exp(shifted).sum())
