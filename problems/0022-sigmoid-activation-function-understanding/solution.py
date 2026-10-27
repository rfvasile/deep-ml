import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    zt = torch.as_tensor(z, dtype=torch.float32)
    r = 1 / (1 + torch.exp(-zt))
    return round(r.item(), 4)
