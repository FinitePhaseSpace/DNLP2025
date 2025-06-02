import torch
import math

def positional_encoding(max_len, d_model):
    """
    Generate positional encoding for a given maximum length and model dimension.

    Args:
        max_len (int): Maximum length of the sequence.
        d_model (int): Dimension of the model.

    Returns:
        torch.Tensor: Positional encoding tensor of shape (max_len, d_model).
    """
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe