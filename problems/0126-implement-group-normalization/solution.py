import numpy as np


def group_normalization(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5
) -> np.ndarray:
    B, C, H, W = X.shape
    assert C % num_groups == 0

    groups = X.reshape((B, num_groups, int(C / num_groups), H, W))

    norm = np.array([])
    for i in range(B):
        for j in range(num_groups):
            group: np.ndarray = groups[i, j, :, :, :]
            mean = group.mean()
            var = group.var(ddof=0)
            group_norm = (group - mean) / np.sqrt(var + epsilon)
            norm = np.append(norm, group_norm)

    norm = norm.reshape((B, C, H, W))
    out = gamma * norm + beta
    return out.astype(np.float32)
