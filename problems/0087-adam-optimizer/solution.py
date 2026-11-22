import numpy as np


def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs one step of the Adam optimization algorithm.
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad**2

    m_ = m / (1 - beta1**t)
    v_ = v / (1 - beta2**t)

    parameter = parameter - learning_rate * m_ / (np.sqrt(v_) + epsilon)
    return np.round(parameter, 5), np.round(m, 5), np.round(v, 5)
