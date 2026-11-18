import numpy as np
from numpy import array, exp, sqrt, swapaxes


def softmax(x):
    """Stable softmax"""
    x = array(x)
    z = x - x.max(axis=-1, keepdims=True)
    return exp(z) / exp(z).sum(axis=-1, keepdims=True)


def compute_qkv(X, W_q, W_k, W_v):
    assert X.ndim == 2
    assert W_q.ndim == 2

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


def self_attention(Q, K, V):
    *B, L, d = Q.shape

    S = (Q @ K.mT) / sqrt(K.shape[-1])
    assert S.shape == (*B, L, L), S.shape

    scores = softmax(S)
    assert scores.shape == (*B, L, L), scores.shape

    out = scores @ V
    assert out.shape == (*B, L, d)

    return out


def split_heads(X, n_heads):
    L, E = X.shape
    X = swapaxes(X, -1, -2)
    X = X.reshape(1, n_heads, E // n_heads, L)
    X = swapaxes(X, -1, -2)
    return X


def multi_head_attention(Q, K, V, n_heads):
    """Use PyTorch convention: N = batch, L = query length, S = key/value length, E = embedding dim"""

    # Unwrap and sanity checks
    L, E = Q.shape
    N = 1
    H = n_heads
    d = E // H

    assert Q.shape[1] % H == 0

    # Add batch dim (N = 1)
    Q = split_heads(Q, H)
    K = split_heads(K, H)
    V = split_heads(V, H)
    assert Q.shape == (N, H, L, d)

    # Attention -> [(L, E)]
    scores = self_attention(Q, K, V)
    assert scores.shape == (N, H, L, d)

    # Reshape to expected format
    scores = swapaxes(scores, 1, 2)
    scores = scores.reshape((L, E))
    return scores


# Test self_attention output shape
np.random.seed(0)
Q = np.random.randn(5, 8)
K = np.random.randn(5, 8)
V = np.random.randn(5, 8)
result = self_attention(Q, K, V)
print(result.shape)
