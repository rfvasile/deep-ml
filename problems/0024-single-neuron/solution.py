import numpy as np
import torch


# ----- PyTorch -----
def single_neuron_model(
    feats_in: list[list[float]],
    labels_in: list[int],
    weights_in: list[float],
    bias: float,
) -> tuple[list[float], float]:
    """
    Simulates a single neuron with sigmoid activation for binary classification.

    Args:
        features: List of feature vectors (each a list of floats)
        labels: List of true binary labels
        weights: Neuron weights (one per feature)
        bias: Neuron bias term

    Returns:
        Tuple of (predicted probabilities rounded to 4 decimal places, MSE rounded to 4 decimal places)
    """
    feats = torch.tensor(feats_in, dtype=torch.float32)
    labels = torch.tensor(labels_in, dtype=torch.float32)
    weights = torch.tensor(weights_in, dtype=torch.float32)

    z = torch.matmul(feats, weights) + bias
    probs = torch.sigmoid(z)
    loss = torch.nn.functional.mse_loss(labels, probs)

    return np.round(probs.tolist(), 4), round(loss.item(), 4)


# ----- Numpy -----
def sigmoid(z: float) -> list[float]:
	return 1 / (1 + np.exp(-z))

def mse_loss(y_true: list[int], y_pred: list[float]) -> float:
	yt = np.array(y_true)
	yp = np.array(y_pred)
	return ((yt - yp) ** 2).sum() / yt.shape[0]

def single_neuron_model2(
	feats_in: list[list[float]],
	labels_in: list[int],
	weights_in: list[float],
	bias: float,
) -> tuple[list[float], float]:
	"""Single neuron network with the sigmoid activation function and MSE loss."""
	feats = np.array(feats_in)
	labels = np.array(labels_in)
	weights = np.array(weights_in)

	z = feats @ weights + bias
	probs = sigmoid(z)
	mse = mse_loss(labels, probs)

	return np.round(probs, 4), np.round(mse, 4)
