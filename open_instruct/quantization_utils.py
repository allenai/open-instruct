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
    quantization_format: Literal["fp8", "fp4", "nvfp4"] = "fp8"
    """the quantization format to use (fp8, fp4, or nvfp4)"""
    quantization_targets: str = "Linear"
    """which layer types to quantize (e.g., 'Linear')"""


def get_quantization_recipe(
    quantization_format: Literal["fp8", "fp4", "nvfp4"], targets: str = "Linear", ignore: Optional[List[str]] = None
):
    try:
        import llmcompressor.modifiers.quantization
    except ImportError as e:
        raise ImportError(
            "llmcompressor is required for quantization. Install it with: pip install llmcompressor"
        ) from e

    if ignore is None:
        ignore = ["lm_head"]

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
    else:
        raise ValueError(f"Unsupported quantization format: {quantization_format}")

    return recipe


def prepare_calibration_dataset(batch_queries: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer):
    calibration_texts = tokenizer.batch_decode(batch_queries, skip_special_tokens=False)

    calibration_dataset = datasets.Dataset.from_dict({"text": calibration_texts})

    return calibration_dataset


def quantize_model_with_batch(
    model: torch.nn.Module,
    quantization_format: Literal["fp8", "fp4", "nvfp4"],
    batch_queries: torch.Tensor,
    tokenizer: transformers.PreTrainedTokenizer,
    targets: str = "Linear",
    max_seq_length: int = 2048,
):
    try:
        import llmcompressor
    except ImportError as e:
        raise ImportError(
            "llmcompressor is required for quantization. Install it with: pip install llmcompressor"
        ) from e

    recipe = get_quantization_recipe(quantization_format, targets=targets)

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
