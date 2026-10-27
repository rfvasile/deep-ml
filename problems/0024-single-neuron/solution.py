import numpy as np


def sigmoid(z: float) -> list[float]:
	return 1 / (1 + np.exp(-z))

def mse_loss(y_true: list[int], y_pred: list[float]) -> float:
	yt = np.array(y_true)
	yp = np.array(y_pred)
	return ((yt - yp) ** 2).sum() / yt.shape[0]

def single_neuron_model(
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
