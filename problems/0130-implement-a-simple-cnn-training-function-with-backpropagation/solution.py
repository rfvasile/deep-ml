# %%
from argparse import Namespace

import numpy as np


def cross_entropy(y_true, y_pred):
    """Implement cross-entropy loss."""
    assert y_true.shape == y_pred.shape, (y_true.shape, y_pred.shape)
    return -(y_true * np.log(y_pred)).sum()


class Conv2d:
    def __init__(
        self,
        W_conv,
        b_conv,
        *,
        lr,
        in_channels,
        out_channels,
        kernel_size,
        in_height,
        in_width,
        padding=0,
        dillation=1,
        stride=1,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Assumed defaults
        out_height = (in_height + 2 * padding - dillation * (kernel_size - 1) - 1) / stride + 1
        out_width = (in_width + 2 * padding - dillation * (kernel_size - 1) - 1) / stride + 1
        assert out_height.is_integer()
        assert out_width.is_integer()
        self.out_height = int(out_height)
        self.out_width = int(out_width)

        self.W_conv = W_conv
        self.b_conv = b_conv
        self.dW_conv = None
        self.input = None
        self.batch_size = None
        self.lr = lr

    def forward(self, batch: np.ndarray):
        self.input = batch
        self.batch_size, C, in_height, in_width = batch.shape
        output = np.zeros((self.batch_size, self.out_channels, self.out_height, self.out_width))

        for b in range(self.batch_size):
            for f_in in range(self.in_channels):
                for f_out in range(self.out_channels):
                    for i in range(0, in_height - self.kernel_size + 1, 1):
                        for j in range(0, in_width - self.kernel_size + 1, 1):
                            patch = batch[b, f_in, i : i + self.kernel_size, j : j + self.kernel_size]
                            assert patch.shape == (self.kernel_size, self.kernel_size), (patch.shape, self.kernel_size)
                            assert self.W_conv[:, :, f_out].shape == (self.kernel_size, self.kernel_size), self.W_conv
                            output[b, f_out, i, j] += np.sum(patch * self.W_conv[:, :, f_out])  # [TODO]

        b_conv = self.b_conv.reshape((1, -1, 1, 1))
        output = output + b_conv
        assert output.shape == (self.batch_size, self.out_channels, self.out_height, self.out_width)
        return output

    def backward(self, grad: np.ndarray):
        assert self.batch_size is not None

        assert grad.shape == (self.batch_size, self.out_channels, self.out_height, self.out_width)
        assert self.W_conv.shape == (self.kernel_size, self.kernel_size, self.out_channels)
        assert self.input is not None, "Must call forward() before backward()"

        # Calculate convolution gradient
        kernel = np.zeros((self.kernel_size, self.kernel_size, self.out_channels))

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                for k in range(self.out_channels):
                    patch = self.input[:, :, i : i + self.out_height, j : j + self.out_width]
                    kernel[i, j, k] = (patch * grad[:, k : k + 1, :, :]).sum()

        # Calculate gradients
        dW_grad = kernel
        db_grad = grad.sum(axis=(0, 2, 3))

        # Update weights
        self.W_conv = self.W_conv - self.lr * dW_grad
        self.b_conv = self.b_conv - self.lr * db_grad

        return dW_grad, db_grad


class FCLayer:
    def __init__(self, W_dense, b_dense, *, out_dim, in_dim, lr):
        super().__init__()
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.lr = lr
        self.W_dense = W_dense
        self.b_dense = b_dense
        self.input = None
        self.batch_dim = None

    def forward(self, batch: np.ndarray):
        self.batch_dim, _ = batch.shape
        self.input = batch

        assert batch.shape == (self.batch_dim, self.in_dim)
        assert self.W_dense.shape == (self.out_dim, self.in_dim)

        y2 = batch @ self.W_dense.T + self.b_dense
        return y2

    def backward(self, grad: np.ndarray):
        assert self.input is not None, "Call forward first"
        assert self.input.shape == (self.batch_dim, self.in_dim)
        assert grad.shape == (self.batch_dim, self.out_dim)  # [Note] (13, 10)

        # Grad w.r.t. weights
        dW_dense = grad.T @ self.input
        db_dense = grad.sum(axis=0)
        assert dW_dense.shape == (self.out_dim, self.in_dim)
        assert db_dense.shape == self.b_dense.shape, (db_dense.shape, self.b_dense.shape)
        assert self.b_dense.shape == (self.out_dim,)

        # Grad w.r.t. input
        dinp = grad @ self.W_dense
        assert dinp.shape == (self.batch_dim, self.in_dim)

        # Update
        self.W_dense = self.W_dense - self.lr * dW_dense
        self.b_dense = self.b_dense - self.lr * db_dense

        return dinp


class Relu:
    def __init__(self):
        super().__init__()
        self.input: np.ndarray | None = None

    def forward(self, batch):
        self.input = batch
        return np.where(batch > 0, batch, 0)

    def backward(self, grad: np.ndarray):
        assert self.input is not None, "Call foward first"
        local_grad = np.where(self.input > 0, 1, 0)
        assert local_grad.shape == grad.shape, (local_grad.shape, grad.shape)
        return local_grad * grad


class Softmax:
    def __init__(self):
        super().__init__()
        self.y_pred = None

    def forward(self, batch):
        exp = batch - batch.max(axis=-1, keepdims=True)
        return np.exp(exp) / np.exp(exp).sum(axis=-1, keepdims=True)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        assert y_pred.shape == y_true.shape
        return y_pred - y_true


class Flatten:
    def __init__(self, batch, channels, height, width):
        super().__init__()
        self.batch = batch
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, batch):
        assert batch.shape == (self.batch, self.channels, self.height, self.width)
        return batch.transpose(0, 2, 3, 1).reshape(batch.shape[0], -1)

    def backward(self, batch):
        assert batch.ndim == 2
        return batch.reshape((self.batch, self.height, self.width, self.channels)).transpose(0, 3, 1, 2)


class Net:
    def __init__(self, W_conv, b_conv, W_dense, b_dense, cfg):
        super().__init__()
        self.activations = []
        self.W_conv = W_conv
        self.b_conv = b_conv
        self.W_dense = W_dense
        self.b_dense = b_dense
        self.out_height = cfg.output_height
        self.out_width = cfg.output_width
        self.kernel_size = cfg.kernel_size
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.lr = cfg.learning_rate
        self.batch_size = cfg.batch_size
        self.num_classes = cfg.num_classes
        self.in_height = cfg.in_height
        self.in_width = cfg.in_width
        self.flattened_size = cfg.output_height * cfg.output_width * cfg.out_channels

        assert W_conv.shape == (cfg.kernel_size, cfg.kernel_size, cfg.out_channels)
        assert b_conv.shape == (cfg.out_channels,)
        assert W_dense.shape == (self.flattened_size, cfg.num_classes)
        assert b_dense.shape == (cfg.num_classes,), (b_dense.shape, cfg.num_classes)
        self.conv1 = Conv2d(
            W_conv=W_conv,
            b_conv=b_conv,
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            kernel_size=cfg.kernel_size,
            in_height=cfg.in_height,
            in_width=cfg.in_width,
            lr=cfg.learning_rate,
        )
        self.relu = Relu()
        self.flatten = Flatten(self.batch_size, self.out_channels, self.out_height, self.out_width)
        self.fc = FCLayer(
            W_dense=W_dense.T,
            b_dense=b_dense,
            out_dim=cfg.num_classes,
            in_dim=self.flattened_size,
            lr=cfg.learning_rate,
        )
        self.softmax = Softmax()

    def forward(self, batch: np.ndarray):
        """Forward pass: conv2d, relu, dense, softmax."""
        B, H, W = batch.shape

        if batch.ndim == 3:
            batch = batch.reshape((B, 1, H, W))

        self.activations = []
        self.activations.append(batch)

        y1 = self.conv1.forward(batch)
        assert y1.shape == (B, self.out_channels, self.out_height, self.out_width)

        z1 = self.relu.forward(y1)
        z1_ = self.flatten.forward(z1)
        assert z1.shape == (B, self.out_channels, self.out_height, self.out_width)
        assert z1_.shape == (B, self.out_channels * self.out_height * self.out_width)

        y2 = self.fc.forward(z1_)
        assert y2.shape == (B, self.num_classes)

        z2 = self.softmax.forward(y2)
        assert z2.shape == (B, self.num_classes)

        return z2

    def backward(self, y_true, y_pred):
        assert y_true.shape == (self.batch_size, self.num_classes), y_true.shape
        assert y_pred.shape == (self.batch_size, self.num_classes), y_pred.shape

        dy5 = self.softmax.backward(y_true=y_true, y_pred=y_pred)
        assert dy5.shape == (self.batch_size, self.num_classes)
        dz1 = self.fc.backward(dy5)

        dz1 = self.flatten.backward(dz1)

        dy2 = self.relu.backward(dz1)
        _, _ = self.conv1.backward(dy2)

        return self.conv1.W_conv, self.conv1.b_conv, self.fc.W_dense.T, self.fc.b_dense


def train_simple_cnn_with_backprop(X, y_true, epochs, learning_rate, kernel_size=3, num_filters=1):
    """
    Trains a simple CNN with one convolutional layer, ReLU activation, flattening, and a dense layer with softmax output using backpropagation.

    Assumes X has shape (n_samples, height, width) for grayscale images and y is one-hot encoded with shape (n_samples, num_classes).

    Parameters:
    X : np.ndarray, input data
    y : np.ndarray, one-hot encoded labels
    epochs : int, number of training epochs
    learning_rate : float, learning rate for weight updates
    kernel_size : int, size of the square convolutional kernel
    num_filters : int, number of filters in the convolutional layer

    Returns:
    W_conv, b_conv, W_dense, b_dense : Trained weights and biases for the convolutional and dense layers
    """
    n_samples, height, width = X.shape
    num_classes = y_true.shape[1]

    # Initialize weights and biases
    W_conv = np.random.randn(kernel_size, kernel_size, num_filters) * 0.01
    b_conv = np.zeros(num_filters)
    output_height = height - kernel_size + 1
    output_width = width - kernel_size + 1
    flattened_size = output_height * output_width * num_filters
    W_dense = np.random.randn(flattened_size, num_classes) * 0.01
    b_dense = np.zeros(num_classes)

    cfg = Namespace(*{})
    cfg.output_height = output_height
    cfg.output_width = output_width
    cfg.kernel_size = kernel_size
    cfg.learning_rate = learning_rate
    cfg.num_classes = num_classes
    cfg.batch_size = 1
    cfg.in_height = height
    cfg.in_width = width
    cfg.in_channels = 1
    cfg.out_channels = num_filters

    net = Net(W_conv, b_conv, W_dense, b_dense, cfg)
    for i in range(epochs):
        # Training
        for n in range(n_samples):
            y_pred = net.forward(X[n : n + 1])

            loss = cross_entropy(y_true[n : n + 1], y_pred)

            W_conv, b_conv, W_dense, b_dense = net.backward(y_true[n : n + 1], y_pred)

    return W_conv, b_conv, W_dense, b_dense
