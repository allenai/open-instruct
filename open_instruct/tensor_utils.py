import torch
import torch.nn.functional as F


def pad_to_length(tensor: torch.Tensor, length: int, pad_value: int | float) -> torch.Tensor:
    """Right-pad a tensor to a specified length along the last dimension."""
    if tensor.size(-1) >= length:
        return tensor
    return F.pad(tensor, (0, length - tensor.size(-1)), value=pad_value)
