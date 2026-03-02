"""
Simple linear model merge by averaging safetensors weights.

This bypasses mergekit's architecture detection, which is useful for
new model architectures that mergekit doesn't support yet.

Usage:
    python -m open_instruct.merge_models \
        --models /path/model1 /path/model2 /path/model3 \
        --output_dir /path/to/output \
        --model_weights 1.0 1.0 1.0
"""

import argparse
import math
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

parser = argparse.ArgumentParser(description="Linear merge of HuggingFace models")
parser.add_argument("--models", nargs="+", required=True, help="Paths to models to merge")
parser.add_argument(
    "--model_weights", nargs="+", type=float, default=None, help="Weights for each model (default: equal weights)"
)
parser.add_argument("--output_dir", required=True, help="Output directory for merged model")
parser.add_argument(
    "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"], help="Output dtype (default: bfloat16)"
)


def get_safetensor_files(model_path: Path) -> list[str]:
    """Get sorted list of safetensor filenames in a model directory."""
    return sorted(f.name for f in model_path.glob("*.safetensors"))


def _load_tensors(model_paths: list[Path], safetensor_file: str) -> list[dict[str, torch.Tensor]]:
    """Load tensors from all models for a single safetensor file."""
    model_tensors = []
    for model_path in model_paths:
        sf_path = model_path / safetensor_file
        with safe_open(str(sf_path), framework="pt", device="cpu") as f:
            tensors = {key: f.get_tensor(key) for key in f.keys()}  # noqa: SIM118 (safe_open is not iterable)
            model_tensors.append(tensors)
    return model_tensors


def _merge_tensors(
    model_tensors: list[dict[str, torch.Tensor]], model_weights: list[float], torch_dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Merge tensors from multiple models using weighted averaging."""
    tensor_names = list(model_tensors[0].keys())
    merged_tensors = {}
    for name in tensor_names:
        merged = torch.zeros_like(model_tensors[0][name], dtype=torch_dtype)
        for i, tensors in enumerate(model_tensors):
            merged += tensors[name].to(torch_dtype) * model_weights[i]
        merged_tensors[name] = merged
    return merged_tensors


def merge_models(
    model_paths: list[Path],
    output_dir: Path,
    model_weights: list[float] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Merge models by weighted averaging of their safetensors weights."""
    model_weights = model_weights or [1.0] * len(model_paths)

    if len(model_weights) != len(model_paths):
        raise ValueError(f"Number of weights ({len(model_weights)}) must match number of models ({len(model_paths)})")

    total_weight = sum(model_weights)
    if math.isclose(total_weight, 0.0):
        raise ValueError("Weights must not sum to zero")
    model_weights = [w / total_weight for w in model_weights]

    logger.info(f"Merging {len(model_paths)} models with weights: {model_weights}")

    safetensor_files = get_safetensor_files(model_paths[0])
    logger.info(f"Found {len(safetensor_files)} safetensor files")

    # Verify all models have identical safetensor file sets
    for model_path in model_paths[1:]:
        other_files = get_safetensor_files(model_path)
        if other_files != safetensor_files:
            raise ValueError(f"Model {model_path} has different safetensor files: {other_files} vs {safetensor_files}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for safetensor_file in tqdm(safetensor_files, desc="Processing files"):
        model_tensors = _load_tensors(model_paths, safetensor_file)
        merged_tensors = _merge_tensors(model_tensors, model_weights, dtype)
        save_file(merged_tensors, str(output_dir / safetensor_file))

    # Copy config files from first model
    config_files = ["config.json", "generation_config.json", "model.safetensors.index.json"]
    for f in config_files:
        src = model_paths[0] / f
        if src.exists():
            shutil.copy(src, output_dir / f)
            logger.info(f"Copied {f}")

    # Copy tokenizer files from first model
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ]
    for f in tokenizer_files:
        src = model_paths[0] / f
        if src.exists():
            shutil.copy(src, output_dir / f)
            logger.info(f"Copied {f}")

    logger.info(f"Merge complete! Output at: {output_dir}")


def main() -> None:
    args = parser.parse_args()
    merge_models(
        model_paths=[Path(p) for p in args.models],
        output_dir=Path(args.output_dir),
        model_weights=args.model_weights,
        dtype=getattr(torch, args.dtype),
    )


if __name__ == "__main__":
    main()
