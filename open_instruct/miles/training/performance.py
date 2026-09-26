"""Explicit rates from global token counts and elapsed phase time."""

import math


def training_rates(model_tokens, active_tokens, seconds, gpus):
    """Counts are global across trainer ranks, not rank-local or FLOP estimates."""
    if not all(math.isfinite(v) for v in (model_tokens, active_tokens, seconds, gpus)):
        raise ValueError("Training rate inputs must be finite")
    if seconds <= 0 or gpus <= 0 or active_tokens < 0 or model_tokens < active_tokens:
        raise ValueError("Training rates require positive time/GPUs and valid global token counts")
    return {
        "model_tokens_per_second": model_tokens / seconds,
        "model_tokens_per_gpu_second": model_tokens / seconds / gpus,
        "active_response_tokens_per_gpu_second": active_tokens / seconds / gpus,
    }
