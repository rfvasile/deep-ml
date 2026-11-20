import numpy as np


def softmax(X):
    exp = X - X.max(axis=-1, keepdims=True)
    return np.exp(exp) / np.exp(exp).sum(axis=-1, keepdims=True)


def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
    """
    Compute Query (Q), Key (K), and Value (V) matrices.
    """
    # (n,I) @ (I, O) -> (n, O)
    return X @ W_q, X @ W_k, X @ W_v


def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute masked self-attention.
    """
    # (n, O) @ (n, O).mT -> (n, n)
    scores = (Q @ K.mT) / np.sqrt(K.shape[-1])

    # (n, n) @ (n, n)
    masked = scores + mask
    probs = softmax(masked)

    # (n, n) @ (n, O)
    out = probs @ V

    return out
