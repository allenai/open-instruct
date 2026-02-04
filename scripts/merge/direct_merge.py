"""
Simple linear model merge by averaging safetensors weights.

This bypasses mergekit's architecture detection, which is useful for
new model architectures that mergekit doesn't support yet.

Usage:
    python scripts/merge/linear_merge.py \
        --models /path/model1 /path/model2 /path/model3 \
        --output_dir /path/to/output \
        --weights 1.0 1.0 1.0
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def get_safetensor_files(model_path: str) -> list[str]:
    """Get list of safetensor files in model directory."""
    model_path = Path(model_path)
    files = list(model_path.glob("*.safetensors"))
    return sorted([f.name for f in files])


def merge_models(
    model_paths: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    dtype: str = "bfloat16",
) -> None:
    """Merge models by averaging their weights."""
    if weights is None:
        weights = [1.0] * len(model_paths)

    if len(weights) != len(model_paths):
        raise ValueError(f"Number of weights ({len(weights)}) must match number of models ({len(model_paths)})")

    # Normalize weights
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Weights must not sum to zero")
    weights = [w / total_weight for w in weights]

    print(f"Merging {len(model_paths)} models with weights: {weights}")

    # Get safetensor files from first model
    safetensor_files = get_safetensor_files(model_paths[0])
    print(f"Found {len(safetensor_files)} safetensor files")

    # Verify all models have the same files
    for model_path in model_paths[1:]:
        other_files = get_safetensor_files(model_path)
        if other_files != safetensor_files:
            raise ValueError(f"Model {model_path} has different safetensor files: {other_files} vs {safetensor_files}")

    os.makedirs(output_dir, exist_ok=True)

    # Determine dtype
    torch_dtype = getattr(torch, dtype)

    # Process each safetensor file
    for sf_file in tqdm(safetensor_files, desc="Processing files"):
        merged_tensors = {}

        # Load tensors from all models
        model_tensors = []
        for model_path in model_paths:
            sf_path = os.path.join(model_path, sf_file)
            with safe_open(sf_path, framework="pt", device="cpu") as f:
                tensors = {key: f.get_tensor(key) for key in f.keys()}
                model_tensors.append(tensors)

        # Get all tensor names from first model
        tensor_names = list(model_tensors[0].keys())

        # Merge each tensor
        for name in tensor_names:
            # Weighted average
            merged = None
            for i, tensors in enumerate(model_tensors):
                tensor = tensors[name].to(torch_dtype)
                if merged is None:
                    merged = tensor * weights[i]
                else:
                    merged = merged + tensor * weights[i]
            merged_tensors[name] = merged

        # Save merged tensors
        output_path = os.path.join(output_dir, sf_file)
        save_file(merged_tensors, output_path)

    # Copy config files from first model
    config_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    ]
    for f in config_files:
        src = os.path.join(model_paths[0], f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"Copied {f}")

    # Copy tokenizer files
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
        src = os.path.join(model_paths[0], f)
        dst = os.path.join(output_dir, f)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"Copied {f}")

    print(f"Merge complete! Output at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Linear merge of HuggingFace models")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Paths to models to merge",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        type=float,
        default=None,
        help="Weights for each model (default: equal weights)",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for merged model",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Output dtype (default: bfloat16)",
    )

    args = parser.parse_args()

    merge_models(
        model_paths=args.models,
        output_dir=args.output_dir,
        weights=args.weights,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
