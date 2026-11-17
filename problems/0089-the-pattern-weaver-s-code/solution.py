import numpy as np

def softmax(values):
    values = np.array(values)
    exponents = values - values.max(axis=1, keepdims=True)
    return np.exp(exponents) / np.exp(exponents).sum(axis=1, keepdims=True)


def pattern_weaver(n, crystal_values, dimension):
    Q, K = n, n
    K_N = dimension
    x = np.array(crystal_values)
    s = np.outer(x, x) / np.sqrt(K_N)  # (Q, K)
    
    assert s.shape == (Q, K)
    a = softmax(s)
    o = a @ x

    return np.round(o, 3)
