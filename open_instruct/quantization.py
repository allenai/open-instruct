"""GPTQ quantization wrapper using llm-compressor."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from llmcompressor.modifiers.quantization.gptq.utils.gptq_quantize import quantize_weight
from llmcompressor.modifiers.quantization.gptq.utils.gptq_wrapper import (
    accumulate_hessian,
    make_empty_hessian,
)
from llmcompressor.pytorch.utils.helpers import align_module_device
from llmcompressor.transformers.utils.helpers import update_offload_parameter

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for GPTQ quantization."""

    enable_gptq: bool = False
    block_size: int = 128
    dampening_frac: float = 0.01
    recalibration_freq: int = 10
    num_calibration_samples: int = 128
    offload_hessians: bool = False


def calibrate_module_with_data(
    module: nn.Module,
    calibration_data: torch.Tensor,
    existing_hessian: Optional[torch.Tensor] = None,
    existing_num_samples: int = 0,
    offload_to_cpu: bool = False,
) -> tuple[torch.Tensor, int]:
    """Calibrate a single module using llm-compressor's accumulate_hessian.

    Args:
        module: The module to calibrate
        calibration_data: Input tensor for calibration [batch_size, seq_len, hidden_size]
        existing_hessian: Existing Hessian to accumulate into (or None to create new)
        existing_num_samples: Number of samples already in existing_hessian
        offload_to_cpu: Whether to store Hessian on CPU to save GPU memory

    Returns:
        (hessian, num_samples) tuple
    """
    device = "cpu" if offload_to_cpu else next(module.parameters()).device

    if existing_hessian is None:
        hessian = make_empty_hessian(module, device=device)
    else:
        hessian = existing_hessian
        if offload_to_cpu:
            hessian = hessian.to(device)

    hessian, num_samples = accumulate_hessian(
        calibration_data,
        module,
        hessian,
        existing_num_samples,
    )

    if offload_to_cpu:
        hessian = hessian.to("cpu")

    return hessian, num_samples


def compress_module_weights(
    module: nn.Module,
    hessian: torch.Tensor,
    num_samples: int,
    quant_args,
    block_size: int = 128,
    dampening_frac: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], float]:
    """Compress a single module's weights using pre-accumulated Hessian.

    Args:
        module: The module to compress
        hessian: Pre-accumulated Hessian matrix
        num_samples: Number of samples used for Hessian
        quant_args: Quantization arguments (from llm-compressor)
        block_size: GPTQ block size
        dampening_frac: Dampening fraction for Hessian regularization

    Returns:
        (quantized_weight, scale, zero_point, g_idx, loss) tuple
    """
    with torch.no_grad(), align_module_device(module):
        device = next(module.parameters()).device
        hessian = hessian.to(device)

        loss, quantized_weight, scale, zero_point, g_idx = quantize_weight(
            module=module,
            quant_args=quant_args,
            hessians_dict={module: hessian},
            blocksize=block_size,
            percdamp=dampening_frac,
        )

    return quantized_weight, scale, zero_point, g_idx, loss


def store_quantized_weights(
    module: nn.Module,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    g_idx: Optional[torch.Tensor],
) -> None:
    """Store quantized weights as buffers on the module.

    Args:
        module: Module to store quantized weights on
        quantized_weight: Quantized int8 weights
        scale: Per-group scales
        zero_point: Per-group zero points
        g_idx: Optional permutation indices for activation ordering
    """
    update_offload_parameter(module, "weight_quantized", quantized_weight)
    update_offload_parameter(module, "weight_scale", scale)
    update_offload_parameter(module, "weight_zero_point", zero_point)
    if g_idx is not None:
        update_offload_parameter(module, "weight_g_idx", g_idx)


def is_quantizable_layer(name: str, module: nn.Module) -> bool:
    """Check if a layer should be quantized.

    Args:
        name: Module name in the model
        module: Module instance

    Returns:
        True if module should be quantized
    """
    if not isinstance(module, nn.Linear):
        return False
    if isinstance(module, nn.Embedding):
        return False
    if "lm_head" in name:
        return False
    return True
