# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Sibling training script for the GENERATIVE value model.

This script is a thin wrapper around ``open_instruct.grpo_fast`` that adds a *second* vLLM pool
hosting a generative value model, runs it at SAE (or fixed-chunk) segment boundaries, feeds the
per-segment scores back to the policy trainer as the value signal, and trains the generative value
model in-place via REINFORCE using the rollout outcome as reward.

The generative value model has its own weights, optimizer, and vLLM pool. Weight sync is driven
through a second NCCL weight-transfer group in parallel with the policy's. See the plan under
"6. Generative value model" for details.

Usage::

    python open_instruct/grpo_fast_genvalue.py \\
        --model_name_or_path ... \\
        --use_generative_value_model \\
        --gen_value_model_name_or_path ... \\
        --gen_value_vllm_num_engines 2 \\
        --gen_value_segmentation sae \\
        --sae_threshold 0.2 \\
        --gen_value_score_min 0 --gen_value_score_max 10 \\
        --gen_value_conditioning gt \\
        --gen_value_reinforce_coef 0.1 \\
        ...
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass

# The sibling script reuses training scaffolding from grpo_fast.py at runtime, but only needs
# grpo_utils and value_model_utils at import time. grpo_fast is heavy (pulls in vLLM) so it's
# imported lazily inside main().
from open_instruct import grpo_utils, value_model_utils

logger = logging.getLogger(__name__)


@dataclass
class GenValueExperimentConfig(grpo_utils.GRPOExperimentConfig):
    """Extended experiment config for the generative-value training script."""

    # Whether to enable the generative value model path (required for this script).
    use_generative_value_model: bool = False
    # Generative value model: its own weights + its own vLLM pool.
    gen_value_model_name_or_path: str | None = None
    gen_value_vllm_num_engines: int = 1
    gen_value_vllm_tensor_parallel_size: int = 1
    # Segmentation: 'sae' uses the policy-logprob-based SAE boundaries (requires --use_sae);
    # 'fixed' queries the gen value model every `gen_value_chunk_size` response tokens.
    gen_value_segmentation: str = "sae"
    gen_value_chunk_size: int = 512
    # Generation params for the gen value model's vLLM engine.
    gen_value_max_new_tokens: int = 8
    gen_value_temperature: float = 1.0
    # Score schema.
    gen_value_score_min: float = 0.0
    gen_value_score_max: float = 10.0
    gen_value_allow_cot: bool = False
    # Training coefficients.
    gen_value_learning_rate: float | None = None
    gen_value_reinforce_coef: float = 0.1
    # Conditioning for the gen-value prompt: one of none, gt, correct_demo, rollout_context.
    gen_value_conditioning: str = "none"

    def __post_init__(self):
        super().__post_init__()
        if not self.use_generative_value_model:
            raise ValueError(
                "grpo_fast_genvalue.py requires --use_generative_value_model. "
                "Use grpo_fast.py for runs without a generative value model."
            )
        if self.gen_value_segmentation not in {"sae", "fixed"}:
            raise ValueError(
                f"--gen_value_segmentation must be 'sae' or 'fixed', got {self.gen_value_segmentation!r}."
            )
        if self.gen_value_segmentation == "sae" and not self.use_sae:
            raise ValueError(
                "--gen_value_segmentation=sae requires --use_sae (SAE boundaries come from the policy's vLLM logprobs)."
            )
        if self.gen_value_chunk_size <= 0:
            raise ValueError(f"--gen_value_chunk_size must be > 0, got {self.gen_value_chunk_size}.")
        if self.gen_value_score_max <= self.gen_value_score_min:
            raise ValueError("--gen_value_score_max must be greater than --gen_value_score_min.")
        if self.gen_value_conditioning not in {"none", "gt", "correct_demo", "rollout_context"}:
            raise ValueError(
                f"--gen_value_conditioning must be one of none/gt/correct_demo/rollout_context, "
                f"got {self.gen_value_conditioning!r}."
            )


def segment_rollout(
    response_tokens: list[int],
    response_logprobs: list[float] | None,
    *,
    mode: str,
    sae_threshold: float = 0.2,
    fixed_chunk_size: int = 512,
) -> list[int]:
    """Return the list of boundary positions (response-token indices) at which the generative value
    model should be queried.

    In ``sae`` mode, a boundary is any response token whose probability (exp(logprob)) is below
    ``sae_threshold``. In ``fixed`` mode, boundaries are emitted every ``fixed_chunk_size`` tokens.
    Boundary indices are returned in ascending order and always include a final boundary at the
    end of the rollout so the terminal outcome is scored.
    """
    import math  # noqa: PLC0415

    length = len(response_tokens)
    if length == 0:
        return []
    boundaries: list[int] = []
    if mode == "sae":
        if response_logprobs is None:
            raise ValueError("SAE segmentation requires response_logprobs.")
        log_threshold = math.log(max(sae_threshold, 1e-12))
        for t, lp in enumerate(response_logprobs):
            if lp < log_threshold:
                boundaries.append(t)
    else:  # fixed
        t = fixed_chunk_size
        while t < length:
            boundaries.append(t)
            t += fixed_chunk_size
    if not boundaries or boundaries[-1] != length - 1:
        boundaries.append(length - 1)
    return boundaries


def score_partial_rollout_batch(
    vllm_engines,
    tokenizer,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    score_min: float,
    score_max: float,
    allow_cot: bool,
) -> tuple[list[float | None], list[str]]:
    """Send a batch of partial-rollout scoring prompts to the gen-value vLLM pool.

    Returns (parsed_scores_in_0_1, raw_generations). ``None`` scores indicate parse failures; the
    caller is free to substitute a default (e.g. 0.5) and track ``parse_rate`` as a metric.
    """
    import asyncio  # noqa: PLC0415

    from vllm import SamplingParams  # noqa: PLC0415

    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_new_tokens,
        top_p=1.0,
        logprobs=1,
    )

    async def _score_one(engine_idx: int, prompt: str) -> tuple[str, float | None]:
        engine = vllm_engines[engine_idx % len(vllm_engines)]
        # The LLMRayActor's add_request interface varies; here we go through its OpenAI server when
        # available. Callers should wire this to match their vllm_utils API version.
        ref = engine.add_request.remote(prompt, sampling_params)
        output = await asyncio.to_thread(__import__("ray").get, ref)
        text = output.outputs[0].text if hasattr(output, "outputs") else str(output)
        parsed = value_model_utils.parse_generative_value_score(
            text, score_min=score_min, score_max=score_max, allow_cot=allow_cot
        )
        return text, parsed

    async def _runner():
        tasks = [_score_one(i, p) for i, p in enumerate(prompts)]
        return await asyncio.gather(*tasks)

    results = asyncio.get_event_loop().run_until_complete(_runner())
    raw = [r[0] for r in results]
    scores = []
    for r in results:
        parsed = r[1]
        if parsed is None:
            scores.append(None)
        else:
            rescaled = (parsed - score_min) / max(score_max - score_min, 1e-8)
            scores.append(max(0.0, min(1.0, rescaled)))
    return scores, raw


def main():
    """Entry point: delegate to grpo_fast.main() but with the extended config + hooks.

    The actual bring-up of a second vLLM pool, its weight-sync group, and the per-step scoring
    path all require a small set of patches to grpo_fast that are gated on
    ``use_generative_value_model``. Those patches are kept out of grpo_fast.py to avoid bloating
    the plain PPO/scalar-value path; instead this script imports the pieces it needs and drives
    the training loop itself.
    """
    # TODO(hamish/vip): wire in the second vLLM pool + weight sync group here. The remaining
    # plumbing is intentionally scaffolded for now (see the plan for the full step-by-step).
    raise NotImplementedError(
        "grpo_fast_genvalue.main() is a scaffold. Use the exposed helpers "
        "(segment_rollout, score_partial_rollout_batch, build_generative_value_prompt, "
        "parse_generative_value_score) to drive a run while the full sibling script lands."
    )


if __name__ == "__main__":
    main()
