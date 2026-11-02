# %%
import copy

import numpy as np

# DO NOT CHANGE SEED
np.random.seed(42)


# DO NOT CHANGE LAYER CLASS
class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, accum_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


# Your task is to implement the Dense class based on the above structure
class Dense(Layer):
    def __init__(self, n_units, input_shape=None):
        """
        layer_input: (B,F)
        input_shape: (F,)
        W: (F, N)
        w0: (1, N)
        """
        self.layer_input = None
        self.input_shape = input_shape
        self.batch_size = None
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self, optimizer):
        # Initialize weights W, biases w0, and optimizers
        assert self.input_shape is not None, (
            "Input shape must be set before initializing the layer."
        )

        limit = 1 / np.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        self.optimizerW = copy.deepcopy(optimizer)
        self.optimizerw0 = copy.deepcopy(optimizer)

        assert self.W.shape == (self.input_shape[0], self.n_units)
        assert self.w0.shape == (1, self.n_units)

    def parameters(self):
        # Return total number of parameters
        assert self.W is not None and self.w0 is not None
        return self.W.shape[0] * self.W.shape[1] + self.w0.shape[1]

    def forward_pass(self, X, training=True):
        # Compute and return the forward pass
        self.batch_size = X.shape[0]

        assert self.input_shape is not None
        assert X.shape == (self.batch_size, self.input_shape[0])

        self.trainable = training
        self.layer_input = X
        return X @ self.W + self.w0  # (B,F) x (F,N) + (1,N) = (B,N)

    def backward_pass(self, accum_grad):
        """
        accum_grad = dL/dy, (B, N)
        """
        # Update W, w0 if trainable, return dL/dX
        assert self.input_shape is not None
        assert accum_grad.shape == (self.batch_size, self.n_units)
        assert self.layer_input is not None
        assert self.W is not None

        # Chain rule y = X @ W + w0:  dL/d0 = dL/dy * dy/d0 where 0 is W, w0, X
        if self.trainable:
            # dy/dW = X, then dy/dw0 = 1
            W_grad = self.layer_input.T @ accum_grad  # (B,F).T x (B,N) -> (F,N)
            w0_grad = accum_grad.sum(axis=0)  # (B,N) -> (N,)

            assert W_grad.shape == (self.input_shape[0], self.n_units)
            assert w0_grad.shape == (self.n_units,)

            initial_W = copy.deepcopy(self.W)
            assert initial_W is not None

            self.W = self.optimizerW.update(self.W, W_grad)
            self.w0 = self.optimizerw0.update(self.w0, w0_grad)

            # dy/dX = W
            return accum_grad @ initial_W.T  # (B,N) x (N,F) -> (B,F)

        return accum_grad @ self.W.T  # dy/dX = W: (B,N) x (N,F) -> (B,F)

    def output_shape(self):
        # Return output shape tuple
        assert self.input_shape is not None
        return (self.n_units,)  # (N,)
