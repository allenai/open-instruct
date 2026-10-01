"""Native Megatron GRPO hooks backed by the same registered OI verifier as Core."""

import json
import math
import os
from pathlib import Path

from open_instruct.miles import rewards


async def reward(args, sample, **kwargs):
    """Pass real task reward to the native GRPO estimator without teacher scoring."""
    value = await rewards.score(sample, os.environ["OI_GRPO_REWARD_CONFIG"], args)
    if not math.isfinite(value):
        raise ValueError("Nonfinite verifier reward")
    return value


def post_process(args, samples, **kwargs):
    """Retain the verifier signal, then center it within prompt groups in native GRPO."""
    values = [float(sample.reward) for sample in samples]
    if not values or not all(map(math.isfinite, values)):
        raise ValueError("Missing or nonfinite verifier rewards")
    if len(samples) % args.n_samples_per_prompt:
        raise ValueError("Incomplete GRPO prompt group")
    path = Path(os.environ["OI_GRPO_OUTPUT"]) / "verifier-rewards.jsonl"
    with path.open("a") as stream:
        for sample, value in zip(samples, values, strict=True):
            if getattr(sample, "teacher_log_probs", None) is not None:
                raise ValueError("Teacher scores must be absent from verifier GRPO")
            stream.write(
                json.dumps(
                    {
                        "sample_index": sample.index,
                        "group_index": getattr(sample, "group_index", None),
                        "tokens": sample.tokens,
                        "response_length": sample.response_length,
                        "response": sample.response,
                        "reward": value,
                        "metadata": sample.metadata,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
    groups = {}
    for index, sample in enumerate(samples):
        group = getattr(sample, "group_index", None)
        groups.setdefault(group if group is not None else index // args.n_samples_per_prompt, []).append(index)
    centered = [0.0] * len(samples)
    for indices in groups.values():
        if len(indices) != args.n_samples_per_prompt:
            raise ValueError("GRPO requires complete fixed-fanout prompt groups")
        prefixes = [samples[i].tokens[: len(samples[i].tokens) - samples[i].response_length] for i in indices]
        if any(prefix != prefixes[0] for prefix in prefixes):
            raise ValueError("GRPO prompt group contains different tokenized prompts")
        mean = sum(values[i] for i in indices) / len(indices)
        denominator = 1.0
        if args.grpo_std_normalization:
            denominator = math.sqrt(sum((values[i] - mean) ** 2 for i in indices) / (len(indices) - 1)) + 1e-6
        for i in indices:
            centered[i] = (values[i] - mean) / denominator
    return values, centered
