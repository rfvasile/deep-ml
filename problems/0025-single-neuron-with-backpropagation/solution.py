import numpy as np


def sigmoid(net: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-net))


def mse_loss(yt: np.ndarray, yp: np.ndarray) -> np.ndarray:
    if len(yt) != len(yp):
        raise ValueError("Length mismatch")

    return ((yt - yp) ** 2).sum() / len(yp)


def train_neuron(
    features: np.ndarray,
    labels: np.ndarray,
    initial_weights: np.ndarray,
    initial_bias: float,
    lr: float,
    epochs: int,
) -> tuple[np.ndarray, float, list[float]]:
    """Training loop including: sigmoid activation, loss and backpropagation."""

    updated_weights = initial_weights
    updated_bias = initial_bias
    mse_values = []
    for _ in range(epochs):
        # Forward pass
        net = features @ updated_weights + updated_bias
        logits = sigmoid(net)
        loss = mse_loss(labels, logits)
        mse_values.append(loss.item())

        # Backward pass
        dNetdz = logits * (1 - logits)
        dLdx = ((2 * (logits - labels) * dNetdz) @ features) / len(labels)
        dLdb = (2 * (logits - labels) @ dNetdz) / len(labels)

        # Update weights
        updated_weights = updated_weights - lr * dLdx
        updated_bias = updated_bias - lr * dLdb

    return (
        np.round(updated_weights, 4),
        np.round(updated_bias, 4),
        np.round(mse_values, 4),
    )
