# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch


def pack_to_bytes(x: torch.LongTensor, elem_bits: int) -> torch.ByteTensor:
    """Pack integer tensor values into byte tensors."""
    assert 1 <= elem_bits <= 64, "elem_bits must be between 1 and 64"

    mask = (1 << elem_bits) - 1
    x = x & mask

    bit_positions = torch.arange(elem_bits - 1, -1, -1, device=x.device)
    bits = ((x.unsqueeze(-1) >> bit_positions) & 1).to(torch.uint8)

    original_shape = x.shape
    bits = bits.view(*original_shape[:-1], -1)

    total_bits = bits.size(-1)
    pad_length = (8 - (total_bits % 8)) % 8
    if pad_length > 0:
        bits = torch.nn.functional.pad(bits, (0, pad_length))
    bits = bits.contiguous()

    bits = bits.view(*original_shape[:-1], -1, 8)
    power = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=x.device)
    return (bits * power).sum(dim=-1).to(torch.uint8)


def unpack_from_bytes(bytes_tensor: torch.ByteTensor, elem_bits: int, original_num_elements: int) -> torch.LongTensor:
    """Unpack byte tensors into integer tensor values."""
    assert 1 <= elem_bits <= 64, "elem_bits must be between 1 and 64"
    assert original_num_elements >= 0, "original_num_elements must be non-negative"

    total_bits_needed = original_num_elements * elem_bits
    original_shape = bytes_tensor.shape
    total_bits_available = original_shape[-1] * 8
    if total_bits_needed > total_bits_available:
        raise ValueError(
            f"Need {total_bits_needed} bits for {original_num_elements} elems, only have {total_bits_available}."
        )

    bit_positions = torch.arange(7, -1, -1, device=bytes_tensor.device)
    bits = ((bytes_tensor.unsqueeze(-1) >> bit_positions) & 1).to(torch.uint8)
    bits_flat = bits.view(*original_shape[:-1], -1)
    bits_needed = bits_flat[..., :total_bits_needed]

    new_shape = list(original_shape[:-1]) + [original_num_elements, elem_bits]
    bits_needed = bits_needed.contiguous().view(*new_shape)

    powers = 2 ** torch.arange(elem_bits - 1, -1, -1, device=bits_needed.device)
    return (bits_needed * powers).sum(dim=-1).long()
