# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import torch

from open_instruct.distillkit.compression.bitpack import pack_to_bytes, unpack_from_bytes
from open_instruct.distillkit.compression.config import DistributionQuantizationConfig, SpecialTerm


def _work_dtype(*inputs: torch.Tensor | None) -> torch.dtype:
    for x in inputs:
        if x is not None and x.dtype == torch.float64:
            return torch.float64
    return torch.float32


def _solve_least_squares(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    work_dtype = _work_dtype(a, b)
    u, s, vh = torch.linalg.svd(a.to(work_dtype), full_matrices=False)
    tol = 1e-5
    s_pinv = torch.zeros_like(s)
    s_pinv[s > tol] = 1 / s[s > tol]
    uh_b = u.transpose(-1, -2) @ b.to(work_dtype)
    s_pinv_uh_b = s_pinv.unsqueeze(-1) * uh_b
    return vh.transpose(-1, -2) @ s_pinv_uh_b


def polynomial_terms(
    terms: list[SpecialTerm | int], t: int, dtype: torch.dtype, device: torch.device, normalize_t: bool
) -> torch.Tensor:
    if normalize_t:
        pts = torch.linspace(0, 1, steps=t, dtype=dtype, device=device)
    else:
        pts = torch.arange(t, dtype=dtype, device=device)
    return torch.stack([pts**i if isinstance(i, int) else getattr(torch, i.value)(pts) for i in terms], dim=-1)


def fit_polynomial(
    values: torch.Tensor, terms: list[SpecialTerm | int], dtype: torch.dtype, normalize_t: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = _work_dtype(values)
    x = polynomial_terms(terms, values.shape[-1], dtype=work_dtype, device=values.device, normalize_t=normalize_t)
    while len(x.shape) < len(values.shape):
        x = x.unsqueeze(0)
    y = values.unsqueeze(-1)
    coeffs = _solve_least_squares(x, y).squeeze(-1)
    coeffs_final = coeffs.to(dtype)
    approx = torch.sum(x.to(dtype) * coeffs_final.unsqueeze(-2), dim=-1).to(work_dtype)
    return coeffs_final, values - approx.squeeze(-1)


def _get_quantize_range(element_bits: int) -> tuple[int, float]:
    if element_bits == 1:
        return 0, 1
    return -(2 ** (element_bits - 1)), (2 ** (element_bits - 1)) - 1.0


def _get_quantize_scale_factors(values: torch.Tensor, element_bits: int) -> tuple[torch.Tensor, float]:
    max_abs_val = torch.amax(torch.abs(values), dim=-1, keepdim=True)
    max_quant_abs = 1.0 if element_bits == 1 else 2 ** (element_bits - 1)
    return max_abs_val, max_quant_abs


def error_diffuse_float(
    values: torch.Tensor, out_dtype: torch.dtype, error_buffer: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    work_dtype = _work_dtype(values, error_buffer)
    values = values.to(work_dtype)
    if error_buffer is None:
        error_buffer = torch.zeros_like(values[..., 0])
    out_values = torch.zeros_like(values, dtype=out_dtype)
    for i in range(values.shape[-1]):
        current = values[..., i] + error_buffer
        q = current.to(out_dtype)
        error_buffer = current - q.to(work_dtype)
        out_values[..., i] = q
    return out_values, error_buffer


def quantize_naive(
    values: torch.Tensor, element_bits: int, scale_dtype: torch.dtype, _error_buffer: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor]:
    work_dtype = _work_dtype(values)
    values = values.to(work_dtype)
    max_abs_val, max_quant_abs = _get_quantize_scale_factors(values, element_bits)
    scale_factor = torch.where(max_abs_val == 0, torch.ones_like(max_abs_val), max_abs_val / max_quant_abs)
    scale_factor = scale_factor.to(scale_dtype).to(values.dtype)
    scaled_vals = values / scale_factor
    quant_min, quant_max = _get_quantize_range(element_bits)
    quantized_values = (torch.round(scaled_vals).clamp(quant_min, quant_max) - quant_min).to(torch.long)
    return scale_factor, quantized_values, torch.zeros_like(scaled_vals[..., 0])


def dequantize(quantized_values: torch.LongTensor, scale: torch.Tensor, element_bits: int) -> torch.Tensor:
    if element_bits == 1:
        return (quantized_values.to(torch.float32) * 2.0 - 1.0) * scale
    quant_min, _ = _get_quantize_range(element_bits)
    return (quantized_values + quant_min).to(torch.float32) * scale


def compress_monotonic_logprobs(logprobs: torch.Tensor, config: DistributionQuantizationConfig) -> torch.ByteTensor:
    if config.delta_encoding:
        deltas = logprobs.to(_work_dtype(logprobs))[..., 1:] - logprobs.to(_work_dtype(logprobs))[..., :-1]
        deltas = torch.cat([logprobs[..., :1], deltas], dim=-1)
        if config.error_diffusion:
            logprobs, _ = error_diffuse_float(deltas, logprobs.dtype, error_buffer=None)
        else:
            logprobs = deltas.to(logprobs.dtype)

    chunks = []
    if config.exact_k > 0:
        exact_values = logprobs[..., : config.exact_k].to(config.exact_dtype.dtype())
        chunks.append(exact_values.view(torch.uint8).reshape(*logprobs.shape[:-1], -1))

    if config.polynomial_terms:
        approx_values = logprobs[..., config.exact_k : config.k]
        coeffs, residual = fit_polynomial(
            approx_values, config.polynomial_terms, dtype=config.term_dtype.dtype(), normalize_t=config.normalize_t
        )
        coeff_bytes = coeffs.to(config.term_dtype.dtype()).view(torch.uint8).reshape(*logprobs.shape[:-1], -1)
        chunks.append(coeff_bytes)
    else:
        residual = logprobs[..., config.exact_k : config.k]

    cur_index = 0
    for bin_ in config.residual_bins:
        values = residual[..., cur_index : cur_index + bin_.num_elements]
        scale, scaled, _ = quantize_naive(values, bin_.element_bits, bin_.scale_dtype.dtype(), None)
        scale_bytes = scale.to(bin_.scale_dtype.dtype()).view(torch.uint8).reshape(*logprobs.shape[:-1], -1)
        chunks.append(scale_bytes)
        packed = pack_to_bytes(scaled, bin_.element_bits).reshape(*logprobs.shape[:-1], -1)
        chunks.append(packed)
        cur_index += bin_.num_elements

    return torch.cat(chunks, dim=-1)


def decompress_monotonic_logprobs(
    bytes_: torch.ByteTensor, config: DistributionQuantizationConfig, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    device = bytes_.device
    if out_dtype is None:
        out_dtype = config.exact_dtype.dtype()

    if config.exact_k > 0:
        exact_dtype_torch = config.exact_dtype.dtype()
        bytes_per_exact = config.exact_dtype.bit_width() // 8
        exact_bytes = config.exact_k * bytes_per_exact
        exact_part = bytes_[..., :exact_bytes].contiguous()
        exact_values = exact_part.view(dtype=exact_dtype_torch).reshape(*bytes_.shape[:-1], config.exact_k)
        remaining_bytes = bytes_[..., exact_bytes:]
    else:
        exact_values = torch.empty((*bytes_.shape[:-1], 0), dtype=out_dtype, device=device)
        remaining_bytes = bytes_

    if config.polynomial_terms:
        terms_count = len(config.polynomial_terms)
        coeff_bytes = terms_count * (config.term_dtype.bit_width() // 8)
        coeff_part = remaining_bytes[..., :coeff_bytes].contiguous()
        coeffs = coeff_part.view(dtype=config.term_dtype.dtype()).reshape(*remaining_bytes.shape[:-1], terms_count)
        remaining_bytes = remaining_bytes[..., coeff_bytes:]
    else:
        coeffs = None

    residuals = []
    for bin_ in config.residual_bins:
        scale_bytes = bin_.scale_dtype.bit_width() // 8
        scale_part = remaining_bytes[..., :scale_bytes].contiguous()
        scale = scale_part.view(dtype=bin_.scale_dtype.dtype()).reshape(*remaining_bytes.shape[:-1], 1)
        remaining_bytes = remaining_bytes[..., scale_bytes:]

        packed_bits = bin_.num_elements * bin_.element_bits
        packed_bytes = (packed_bits + 7) // 8
        packed_part = remaining_bytes[..., :packed_bytes].contiguous()
        remaining_bytes = remaining_bytes[..., packed_bytes:]

        elements = unpack_from_bytes(packed_part, bin_.element_bits, bin_.num_elements)
        residuals.append(dequantize(elements, scale, bin_.element_bits))

    approx_terms = config.k - config.exact_k
    if residuals:
        residual = torch.cat(residuals, dim=-1)
        if residual.shape[-1] < approx_terms:
            residual = torch.nn.functional.pad(residual, (0, approx_terms - residual.shape[-1]))
    else:
        residual = torch.zeros((*bytes_.shape[:-1], approx_terms), dtype=out_dtype, device=device)

    if coeffs is not None and config.polynomial_terms:
        x = polynomial_terms(
            terms=config.polynomial_terms,
            t=approx_terms,
            dtype=config.term_dtype.dtype(),
            device=device,
            normalize_t=config.normalize_t,
        )
        fit = torch.sum(x.to(coeffs.device, coeffs.dtype) * coeffs.unsqueeze(-2), dim=-1)
        approx_values = fit + residual.to(out_dtype)
    else:
        approx_values = residual.to(out_dtype)

    logprobs = torch.cat([exact_values.to(out_dtype), approx_values], dim=-1)
    if config.delta_encoding:
        logprobs = torch.cumsum(logprobs.float(), dim=-1).to(out_dtype)
    return logprobs
