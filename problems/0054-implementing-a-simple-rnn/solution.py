import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    """
    Wx is (H,D), Wh is (H,H). Each step maps x (D,) to h (H,).
    """
    (x, h, w1, w2) = (
        np.array(input_sequence),
        np.array(initial_hidden_state),
        np.array(Wx),
        np.array(Wh),
    )
    for t in x:
        h = np.tanh(w1 @ t + w2 @ h + b)

    return np.round(h, 4).tolist()
