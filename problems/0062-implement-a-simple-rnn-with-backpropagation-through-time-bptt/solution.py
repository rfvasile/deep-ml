# %%
import numpy as np


def mse(y_true, y_pred):
    assert y_true.shape == y_pred.shape, (
        f"y_true.shape() -> {y_true.shape}, y_pred.shape() -> {y_pred.shape}"
    )
    return ((y_pred - y_true) ** 2).sum() / len(y_pred)


def tanh(x):
    return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)


class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the RNN with random weights and zero biases.
        """
        self.hidden_size = hidden_size
        self.W_xh = np.random.randn(hidden_size, input_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(output_size, hidden_size) * 0.01
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))

        self.h_pred = np.array([])
        self.y_pred = np.array([])

    def forward(self, x):
        """
        Forward pass through the RNN for a given sequence of inputs.
        Input:
          x (T, input_size)
        Return:
          out (T, output_size, 1)
        """
        h = np.zeros((self.hidden_size, 1))
        T, _ = x.shape
        O, _ = self.b_y.shape

        self.y_pred = np.array([])
        self.h_pred = np.array([])
        for t in x:
            t = t.reshape((len(t), 1))
            # (H,I) @ (I,1) + (H,H) @ (H,1) + (H,1)
            # (H,1) + (H,1) + (H,1) -> (H,1)
            h = tanh(self.W_xh @ t + self.W_hh @ h + self.b_h)

            # (O, H) @ (H, 1) + (O, 1) -> (O, 1)
            y = self.W_hy @ h + self.b_y
            self.h_pred = np.append(self.h_pred, h)
            self.y_pred = np.append(self.y_pred, y)

        self.y_pred = self.y_pred.reshape((T, O, 1))
        self.h_pred = self.h_pred.reshape((T, self.hidden_size))

        return self.y_pred

    def backward(self, x, y_true, learning_rate):
        """
        Backpropagation through time to adjust weights based on error gradient.
        Input:
          y (T, output_size)
        """

        T, I = x.shape
        O, _ = self.b_y.shape
        H = self.hidden_size
        assert self.h_pred.shape == (T, H)
        assert x.shape == (T, I)
        assert y_true.shape == (T, O)

        dhprev, dW_hy, db_y, dW_xh, dW_hh, db_h = (
            np.zeros((H, 1)),
            np.zeros((O, H)),
            np.zeros((O, 1)),
            np.zeros((H, I)),
            np.zeros((H, H)),
            np.zeros((H, 1)),
        )
        for i, t in zip(reversed(range(len(x))), x[::-1, :]):
            t = t.reshape((len(t), 1))  # (I, 1)

            # Loss gradient
            dLd_y = self.y_pred[i] - y_true[i].reshape((O, 1))
            assert dLd_y.shape == (O, 1)

            # Other gradients
            dW_hy += np.outer(dLd_y, self.h_pred[i])
            db_y += dLd_y
            assert dW_hy.shape == (O, H)
            assert db_y.shape == (O, 1)

            d_h = self.W_hy.T @ dLd_y + dhprev
            dtanh = d_h * (1 - self.h_pred[i][:, np.newaxis] ** 2)
            dhprev = self.W_hh.T @ dtanh
            assert d_h.shape == (H, 1)
            assert dtanh.shape == (H, 1)
            assert dhprev.shape == (H, 1)

            dW_xh += np.outer(dtanh, t)
            assert dW_xh.shape == (H, I)

            h_pred = self.h_pred[i - 1] if i > 0 else np.zeros((H,))
            dW_hh += np.outer(dtanh, h_pred)
            db_h += dtanh
            assert dW_hh.shape == (H, H)
            assert db_h.shape == (H, 1)

        self.W_xh = self.W_xh - learning_rate * dW_xh
        self.W_hh = self.W_hh - learning_rate * dW_hh
        self.W_hy = self.W_hy - learning_rate * dW_hy
        self.b_h = self.b_h - learning_rate * db_h
        self.b_y = self.b_y - learning_rate * db_y
