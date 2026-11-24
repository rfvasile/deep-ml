import numpy as np


def softplus(z: np.ndarray):
    """
    Softplus is always positive, and increases/decreases slowly.
    This is important during training for stability.
    """
    return np.log(1 + np.exp(z))


def top_k(H: np.ndarray, k: int):
    return np.argsort(H, axis=-1)[:, -k:][:, ::-1]


def softmax(z: np.ndarray):
    exp = z - z.max(axis=-1, keepdims=True)
    return np.exp(exp) / np.exp(exp).sum(axis=-1, keepdims=True)


def noisy_topk_gating(X: np.ndarray, W_g: np.ndarray, W_noise: np.ndarray, N: np.ndarray, k: int) -> np.ndarray:
    """
        X: Input data, shape (batch_size, features)
        W_g: Gating weight matrix, shape (features, num_experts)
        W_noise: Noise weight matrix, shape (features, num_experts)
        N: Noise samples, shape (batch_size, num_experts)
        k: Number of experts to keep per example
    Returns:
        Gating probabilities, shape (batch_size, num_experts)
    """
    # Router scores: the noise enables stable training
    z_g = X @ W_g
    z_noise = X @ W_noise

    # Noisy logits: softplus enables alternative experts to learn
    H = z_g + N * softplus(z_noise)

    # Sparsity: apply MoE to k largest noisy logits, set others to -inf
    indices = top_k(H, k)
    out = np.full_like(H, -np.inf, dtype=float)
    np.put_along_axis(out, indices, np.take_along_axis(H, indices, axis=-1), axis=-1)

    # Gating weights: ignore discarded experts output, and consider first k
    probs = softmax(out)
    return np.round(probs, 3)
