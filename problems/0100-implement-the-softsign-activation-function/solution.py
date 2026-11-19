import numpy as np


def softsign(x):
    val = x / (1 + np.abs(x))
    return round(val, 4)
