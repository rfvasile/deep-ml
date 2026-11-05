import numpy as np


def adam_optimizer(
    f,
    grad,
    x0,
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    num_iterations=10,
):
    t = 0
    mt = np.zeros_like(x0)
    vt = np.zeros_like(x0)

    while t < num_iterations:
        t = t + 1
        gt = grad(x0)

        # Compute first and second moments
        mt = beta1 * mt + (1 - beta1) * gt
        vt = beta2 * vt + (1 - beta2) * gt**2

        # Compute bias corrected first and second moments
        _mt = mt / (1 - beta1**t)
        _vt = vt / (1 - beta2**t)

        # Update parameters
        x0 = x0 - learning_rate * _mt / (np.sqrt(_vt) + epsilon)

    return x0
