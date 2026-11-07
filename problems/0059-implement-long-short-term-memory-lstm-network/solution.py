# %%
# c(t-1), h(t-1), x(t) -> c(t), h(t)
# forget gate f(t):
# - discard from c(t-1) what is not needed
# - uses h(t-1) and x(t)
# - outputs: x in [0, 1] for each number in c(t-1) (0 = forget, 1 = keep)

# Input gate i(t):
# - decides what new information to store in c(t)
# - consists of two parts:
#  1. sigmoid layer: decides which values to update
#  2. tanh layer: creates a vector of new candidate values, c~(t), that could be added to the state
#
# Cell state update c(t):
# - updates old cell state c(t-1) into new cell state c(t)
# - multiplies the old state by f(t), then adds i(t) * c~(t)
#
# Output gate o(t):
# - decides what to output based on cell state c(t)
# - uses sigmoid fn to decide which parts of the cell state to output
# - then multiplies it by tanh(c(t)) to push the values to be between -1 and 1

# %%

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x: float) -> float:
    """
    Implements the Tanh (hyperbolic tangent) activation function.

    Args:
            x (float): Input value

    Returns:
            float: The tanh of the input
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


class LSTM:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

        self.bc = np.zeros((hidden_size, 1))
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def forward(self, x, initial_hidden_state, initial_cell_state):
        """
        Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
        """

        h = initial_hidden_state
        c = initial_cell_state
        out = []

        x = np.asarray(x).reshape(len(x), -1, 1)
        for t in x:
            inp = np.concatenate((h, t))

            # Forget gate
            # (H,H + D) @ (H + D,1) + (H,1) -> (H,1)
            f = sigmoid(self.Wf @ inp + self.bf)

            # Input gate
            i = sigmoid(self.Wi @ inp + self.bi)
            c_cand = tanh(self.Wc @ inp + self.bc)

            # Cell state
            c = f * c + i * c_cand

            # Output
            o = sigmoid(self.Wo @ inp + self.bo)
            h = o * tanh(c)
            out.append(h)

        return [np.round(h, 4) for h in out], h, c
