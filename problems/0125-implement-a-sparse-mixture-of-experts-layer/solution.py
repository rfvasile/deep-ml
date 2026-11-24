import numpy as np


def get_top_k(probs: np.ndarray, k: int):
    return np.argsort(probs, axis=-1)[..., -k:][..., ::-1]


def softmax(z: np.ndarray):
    exp = z - z.max(axis=-1, keepdims=True)
    return np.exp(exp) / np.exp(exp).sum(axis=-1, keepdims=True)


def moe(x: np.ndarray, We: np.ndarray, Wg: np.ndarray, n_experts: int, k: int) -> np.ndarray:
    """
    Args:
        x: Input tensor of shape (n_batch, l_seq, d_model)
        We: Expert weights of shape (n_experts, d_model, d_model)
        Wg: Gating weights of shape (d_model, n_experts)
        n_experts: Number of experts
        k: Number of experts to route each token to
    Returns:
        Output tensor of shape (n_batch, l_seq, d_model)
    """
    B, L, H = x.shape
    _, E = Wg.shape

    ##### Router #####
    # Gating with softmax
    z_g = x @ Wg
    assert z_g.shape == (B, L, E)

    probs = softmax(z_g)
    assert probs.shape == (B, L, E)
    assert np.allclose(probs.sum(axis=-1, keepdims=True), 1.0)

    # Top-k selection: sparsity + renormalize
    indices = get_top_k(probs, k)
    out = np.full_like(probs, -np.inf, dtype=float)
    np.put_along_axis(out, indices, np.take_along_axis(z_g, indices, axis=-1), axis=-1)
    assert out.shape == (B, L, E)

    probs = softmax(out)
    assert probs.shape == (B, L, E)

    ##### Expert calculation #####
    # Expert computation
    o = np.einsum("blh,ehd->bled", x, We)
    assert o.shape == (B, L, E, H)

    # Weight aggregation
    o = np.einsum("bleh,ble->blh", o, probs)
    assert o.shape == (B, L, H)
    return o
