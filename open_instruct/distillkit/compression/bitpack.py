# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch


def pack_to_bytes(x: torch.Tensor, elem_bits: int) -> torch.Tensor:
    """Pack integer tensor values into byte tensors."""
    if not 1 <= elem_bits <= 64:
        raise ValueError("elem_bits must be between 1 and 64")

    # keep only low elem_bits bits from each element before bit-packing
    x = x & ((1 << elem_bits) - 1)

    # expand each element to big-endian bit values
    bits = ((x.unsqueeze(-1) >> torch.arange(elem_bits - 1, -1, -1, device=x.device)) & 1).to(torch.uint8)

    # flatten per-element bits into trailing bitstream
    bits = bits.view(*x.shape[:-1], -1)

    bits = torch.nn.functional.pad(bits, (0, (8 - (bits.size(-1) % 8)) % 8)).contiguous()

    # regroup trailing bitstream into bytes
    bits = bits.view(*x.shape[:-1], -1, 8)
    power = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8, device=x.device)
    return (bits * power).sum(dim=-1).to(torch.uint8)


def unpack_from_bytes(bytes_tensor: torch.Tensor, elem_bits: int, original_num_elements: int) -> torch.Tensor:
    """Unpack byte tensors into integer tensor values."""
    if not 1 <= elem_bits <= 64:
        raise ValueError("elem_bits must be between 1 and 64")
    if original_num_elements < 0:
        raise ValueError("original_num_elements must be non-negative")

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

    bits_needed = bits_needed.contiguous().view(*original_shape[:-1], original_num_elements, elem_bits)

    powers = 2 ** torch.arange(elem_bits - 1, -1, -1, device=bits_needed.device)
    return (bits_needed * powers).sum(dim=-1).long()
