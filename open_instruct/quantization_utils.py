import logging
from dataclasses import dataclass
from typing import List, Literal, Optional

import datasets
import torch
import transformers

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    enable_quantization: bool = False
    """whether to enable quantization of model weights before syncing to vLLM. If disabled, defaults to bf16."""
    quantization_format: Literal["fp8", "fp4", "nvfp4", "gptq"] = "fp8"
    """the quantization format to use (fp8, fp4, nvfp4, or gptq for SmoothQuant+GPTQ)"""
    quantization_targets: str = "Linear"
    """which layer types to quantize (e.g., 'Linear')"""
    smoothquant_strength: float = 0.5
    """alpha used by SmoothQuant when quantization_format='gptq'"""
    gptq_bits: int = 4
    """number of weight bits for GPTQ quantization"""
    gptq_group_size: int = 128
    """group size for GPTQ weight quantization"""
    gptq_actorder: Optional[Literal["static", "descending", "ascending"]] = "static"
    """activation ordering strategy for GPTQ (None disables actorder)"""
    quantization_calibration_samples: int = 256
    """number of prompt sequences to retain for quantization calibration"""


def get_quantization_recipe(
    quantization_format: Literal["fp8", "fp4", "nvfp4", "gptq"],
    targets: str = "Linear",
    ignore: Optional[List[str]] = None,
    smoothquant_strength: float = 0.5,
    gptq_bits: int = 4,
    gptq_group_size: int = 128,
    gptq_actorder: Optional[str] = "static",
):
    try:
        import llmcompressor.modifiers.quantization
    except ImportError as e:
        raise ImportError(
            "llmcompressor is required for quantization. Install it with: pip install llmcompressor"
        ) from e

    if ignore is None:
        ignore = ["lm_head"]
    else:
        ignore = list(ignore)

    target_list = [targets] if isinstance(targets, str) else list(targets)

    if quantization_format == "fp8":
        recipe = llmcompressor.modifiers.quantization.QuantizationModifier(
            targets=targets, scheme="W8A16", ignore=ignore
        )
    elif quantization_format == "fp4":
        recipe = llmcompressor.modifiers.quantization.QuantizationModifier(
            targets=targets, scheme="W8A16", ignore=ignore
        )
    elif quantization_format == "nvfp4":
        recipe = llmcompressor.modifiers.quantization.QuantizationModifier(
            targets=targets, scheme="NVFP4", ignore=ignore
        )
    elif quantization_format == "gptq":
        from compressed_tensors.quantization.quant_scheme import preset_name_to_scheme
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier

        if gptq_bits <= 0:
            raise ValueError("gptq_bits must be a positive integer")

        scheme_name = f"W{gptq_bits}A16"

        try:
            quant_scheme = preset_name_to_scheme(scheme_name, target_list)
        except KeyError as exc:
            raise ValueError(
                f"Unsupported GPTQ bit-width {gptq_bits}. Supported presets include 4 and 8 bits."
            ) from exc

        if quant_scheme.weights is None:
            raise ValueError(f"Preset scheme {scheme_name} does not define weights configuration.")

        quant_scheme.weights.num_bits = gptq_bits
        if quant_scheme.weights.strategy is not None and gptq_group_size is not None:
            quant_scheme.weights.group_size = gptq_group_size

        config_groups = {"group_0": quant_scheme}

        smooth_modifier = SmoothQuantModifier(smoothing_strength=smoothquant_strength, ignore=ignore)
        gptq_modifier = GPTQModifier(
            config_groups=config_groups,
            ignore=ignore,
            actorder=gptq_actorder,
            block_size=gptq_group_size,
        )
        recipe = [smooth_modifier, gptq_modifier]
    else:
        raise ValueError(f"Unsupported quantization format: {quantization_format}")

    return recipe


def prepare_calibration_dataset(batch_queries: List[List[int]], tokenizer: transformers.PreTrainedTokenizer):
    calibration_texts = tokenizer.batch_decode(batch_queries, skip_special_tokens=False)

    calibration_dataset = datasets.Dataset.from_dict({"text": calibration_texts})

    return calibration_dataset


def quantize_model_with_batch(
    model: torch.nn.Module,
    quantization_format: Literal["fp8", "fp4", "nvfp4", "gptq"],
    batch_queries: List[List[int]],
    tokenizer: transformers.PreTrainedTokenizer,
    targets: str = "Linear",
    max_seq_length: int = 2048,
    smoothquant_strength: float = 0.5,
    gptq_bits: int = 4,
    gptq_group_size: int = 128,
    gptq_actorder: Optional[str] = "static",
):
    try:
        import llmcompressor
    except ImportError as e:
        raise ImportError(
            "llmcompressor is required for quantization. Install it with: pip install llmcompressor"
        ) from e

    recipe = get_quantization_recipe(
        quantization_format,
        targets=targets,
        smoothquant_strength=smoothquant_strength,
        gptq_bits=gptq_bits,
        gptq_group_size=gptq_group_size,
        gptq_actorder=gptq_actorder,
    )

    if quantization_format == "gptq":
        logger.info(
            "Applying SmoothQuant(alpha=%.3f) + GPTQ(bits=%d, group_size=%d, actorder=%s)",
            smoothquant_strength,
            gptq_bits,
            gptq_group_size,
            gptq_actorder or "none",
        )

    calibration_dataset = prepare_calibration_dataset(batch_queries, tokenizer)

    logger.info(f"Quantizing model to {quantization_format} using {len(calibration_dataset)} calibration samples...")

    llmcompressor.oneshot(
        model=model,
        dataset=calibration_dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=len(calibration_dataset),
    )

    logger.info(f"Model quantized to {quantization_format}")

    return model
