import numpy as np


def prelu_forward(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Implements the forward pass of PReLU.
    """
    return np.where(x > 0, x, alpha * x)


def prelu_backward(x: np.ndarray, alpha: float, grad_output: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Implements the backward pass of PReLU, computing gradients for both x and alpha.
    """
    grad_x = grad_output * np.where(x > 0, 1, alpha)
    grad_alpha = (grad_output * np.where(x > 0, 0, x)).sum()
    return grad_x, grad_alpha


# %%

class PreluNet:
    def __init__(self, x: np.ndarray, lr: float, alpha):
        self.W_x = np.eye(x.shape[0])
        self.alpha = alpha
        self.y = None
        self.lr = lr

    def prelu_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Implements the forward pass of PReLU.
        """
        self.y = x @ self.W_x.T
        z = np.where(self.y > 0, self.y, self.alpha * self.y)
        return z

    def prelu_backward(self, x) -> tuple[np.ndarray, float]:
        """
        Implements the backward pass of PReLU, computing gradients for both W_x and alpha.
        """
        assert self.y is not None
        d_prelu = np.where(self.y > 0, 1, self.alpha)
        grad_Wx = np.outer(d_prelu, x)  # (O, I)
        grad_alpha = np.where(self.y > 0, 0, self.y)
        grad_alpha = grad_alpha.sum()

        self.W_x = self.W_x - self.lr * grad_Wx
        self.alpha = self.alpha - self.lr * grad_alpha
        return grad_Wx, grad_alpha
