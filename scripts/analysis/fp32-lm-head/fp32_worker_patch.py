"""
FP32 LM head setup for vLLM workers.

This module contains functions that are executed inside vLLM worker processes
via apply_model(). They must be in a separate module (not __main__) to be
picklable for multiprocessing.
"""

import os

import torch

# Ensure the env var is set so patch_vllm_for_fp32_logits uses cache mode
os.environ.setdefault("OPEN_INSTRUCT_FP32_LM_HEAD", "1")


def setup_fp32_in_worker(model: torch.nn.Module) -> str:
    """
    Runs inside each vLLM worker process (via apply_model).

    This function does TWO things:
    1. Patches LogitsProcessor._get_logits to use fp32 weights (using existing implementation)
    2. Sets lm_head._open_instruct_fp32_weight cache on the model

    Both steps are necessary: the patch makes LogitsProcessor LOOK for fp32 weights,
    and the cache provides the actual fp32 weights to use.
    """
    results = []

    # Step 1: Patch LogitsProcessor using the existing implementation
    try:
        from open_instruct.vllm_utils import patch_vllm_for_fp32_logits

        patch_vllm_for_fp32_logits(True)
        results.append("patched LogitsProcessor")
    except Exception as e:
        results.append(f"patch failed: {e}")

    # Step 2: Set fp32 weight cache on model
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        results.append("no lm_head found")
        return "; ".join(results)

    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        results.append("lm_head.weight not a tensor")
        return "; ".join(results)

    # Avoid repeated allocations if called more than once
    cached = getattr(lm_head, "_open_instruct_fp32_weight", None)
    if (
        isinstance(cached, torch.Tensor)
        and cached.dtype == torch.float32
        and cached.device == weight.device
        and cached.shape == weight.shape
    ):
        results.append(f"fp32 cache already exists (shape={weight.shape})")
        return "; ".join(results)

    with torch.no_grad():
        lm_head._open_instruct_fp32_weight = weight.detach().float()
    results.append(f"created fp32 cache (shape={weight.shape}, device={weight.device})")

    return "; ".join(results)
