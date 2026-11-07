import numpy as np

def tanh(x: float) -> float:
	"""
	Implements the Tanh (hyperbolic tangent) activation function.

	Args:
		x (float): Input value

	Returns:
		float: The tanh of the input, rounded to 4 decimal places
	"""
	return np.round((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)), 4)
