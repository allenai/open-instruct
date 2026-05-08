# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Helpers for the PPO / SAE value model used by grpo_fast.py.

The value model itself is built, optimized, and DeepSpeed-managed inside
`PolicyTrainerRayProcess.from_pretrained`; this module provides stateless helpers for:

- building value-conditioning strings from ground truths + sibling rollouts;
- running the value forward with or without between-prompt-and-response conditioning;
- extracting per-token values for the scalar regression head.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

import torch

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

_ROLLOUT_CONTEXT_MAX_TOKENS = 4096
_RUBRIC_CONDITIONING_MAX_TOKENS = 2048
_DEFAULT_ROLLOUT_CONTEXT_NUM_SIBLINGS = 4
_AUTO_ROLLOUT_CONTEXT_NUM_SIBLINGS = -1
_POSTFIX_TEMPLATES = {"expected_accuracy", "rollout_context", "correct_demo"}
_PREFIX_TEMPLATES = {"answer_prefix", "rubrics"}

# Templates that need a ground-truth string to be meaningful.
TEMPLATES_REQUIRING_GT: frozenset[str] = frozenset({"rollout_context", "correct_demo", "rubrics"})
# Templates that need sibling rollouts (decoded responses from the same prompt group).
TEMPLATES_REQUIRING_SIBLINGS: frozenset[str] = frozenset({"rollout_context", "correct_demo"})
# All valid gt_conditioning_template values.
ALL_GT_CONDITIONING_TEMPLATES: frozenset[str] = frozenset(_POSTFIX_TEMPLATES | _PREFIX_TEMPLATES)
# Valid values for --gen_value_conditioning.
GEN_VALUE_CONDITIONING_TYPES: frozenset[str] = frozenset({"none", "gt", "correct_demo", "rollout_context"})
TEMPLATES_USING_HINTS: frozenset[str] = frozenset({"answer_prefix"})


def segment_rollout(
    response_tokens: list[int],
    response_logprobs: list[float] | None,
    *,
    mode: str,
    sae_threshold: float = 0.2,
    fixed_chunk_size: int = 512,
    max_segments: int | None = None,
) -> list[int]:
    """Return boundary positions (response-token indices) at which the gen-value model is queried.

    In ``sae`` mode boundaries are tokens whose probability is below ``sae_threshold``.
    In ``fixed`` mode boundaries are emitted every ``fixed_chunk_size`` tokens.
    The final token is always a boundary.

    When ``max_segments`` is set and the raw boundary count exceeds it, boundaries are
    downsampled to ``max_segments`` by picking evenly spaced entries from the full list.
    """
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
    if max_segments is not None and len(boundaries) > max_segments:
        if max_segments < 1:
            raise ValueError(f"max_segments must be >= 1, got {max_segments}")
        if max_segments == 1:
            boundaries = [length - 1]
        else:
            n = len(boundaries)
            step = (n - 1) / (max_segments - 1)
            kept = [boundaries[round(i * step)] for i in range(max_segments)]
            kept[-1] = length - 1
            boundaries = kept
    return boundaries


def rescale_gen_value_score(parsed: float, score_min: float, score_max: float) -> float:
    """Rescale a raw gen-value score from [score_min, score_max] to [0, 1]."""
    return max(0.0, min(1.0, (parsed - score_min) / max(score_max - score_min, 1e-8)))


def is_postfix_template(template: str) -> bool:
    """Postfix templates are spliced BETWEEN prompt and response (per sub-sequence).

    Prefix templates are prepended to the entire packed sub-sequence.
    """
    return template in _POSTFIX_TEMPLATES


def resolve_num_siblings_to_sample(
    template: str, num_siblings_to_sample: int, num_samples_per_prompt: int
) -> int:
    """Resolve the auto sibling count used by rollout-context-style templates.

    ``correct_demo`` needs access to every other rollout by default so it does not
    drop a successful sibling before choosing the reference demo.
    """
    if num_siblings_to_sample >= 0:
        return num_siblings_to_sample
    if num_siblings_to_sample != _AUTO_ROLLOUT_CONTEXT_NUM_SIBLINGS:
        raise ValueError(f"num_siblings_to_sample must be >= -1, got {num_siblings_to_sample}.")
    if template == "correct_demo":
        return max(0, num_samples_per_prompt - 1)
    return _DEFAULT_ROLLOUT_CONTEXT_NUM_SIBLINGS


def build_conditioning_text(
    template: str, ground_truth: str, siblings: Sequence[dict] | None = None, hint: str | None = None
) -> str:
    """Return the conditioning text to splice for a single sub-sequence.

    The text is inserted between the prompt and the response (postfix templates) or as a prefix to
    the whole sub-sequence (prefix templates). Callers are expected to tokenize the returned
    string and extend position ids accordingly.
    """
    if template == "answer_prefix":
        if hint is not None:
            return f"Here is a hint: {hint}.\n"
        return f"The correct answer is: {ground_truth}\n"
    if template == "expected_accuracy":
        return f"Given the answer is {ground_truth}, Let me compute the expected accuracy of the partial rollout: "
    if template == "rollout_context":
        return _build_rollout_context(ground_truth, siblings or [])
    if template == "correct_demo":
        return _build_correct_demo_context(ground_truth, siblings or [])
    if template == "rubrics":
        return _build_rubric_context(ground_truth)
    raise ValueError(f"Unknown gt_conditioning_template: {template!r}")


def _build_rollout_context(ground_truth: str, siblings: Sequence[dict]) -> str:
    header = "Here are some other attempts at this question:\n"
    suffix = f"Given the answer is {ground_truth}, compute the expected accuracy of the current attempt: "
    if not siblings:
        return header + suffix
    lines: list[str] = []
    budget = _ROLLOUT_CONTEXT_MAX_TOKENS
    sorted_siblings = sorted(siblings, key=lambda s: len(str(s.get("text", ""))))
    for k, s in enumerate(sorted_siblings):
        tag = "CORRECT" if s.get("is_correct") else "INCORRECT"
        text = str(s.get("text", ""))
        line = f"Attempt {k + 1} ({tag}):\n{text}\n"
        approx_tokens = max(1, len(line) // 4)
        if approx_tokens > budget:
            continue
        budget -= approx_tokens
        lines.append(line)
    return header + "".join(lines) + suffix


def _build_correct_demo_context(ground_truth: str, siblings: Sequence[dict]) -> str:
    """Pick ONE sibling (prefer a correct one); if none, return a blank reference."""
    chosen = None
    for s in siblings:
        if s.get("is_correct"):
            chosen = s
            break
    if chosen is None and siblings:
        chosen = siblings[0]
    tag = "CORRECT" if (chosen and chosen.get("is_correct")) else "INCORRECT"
    text = str(chosen.get("text", "")) if chosen else ""
    reference = f"Here is a reference attempt ({tag}):\n{text}\n" if chosen else ""
    return reference + f"Given the answer is {ground_truth}, compute the expected accuracy of the current attempt: "


def _build_rubric_context(ground_truth: str) -> str:
    """Format the rubrics from a JSON ground truth as a value-model conditioning prefix.

    The ``ground_truth`` is the JSON-encoded payload that ``apply_evolving_rubric_reward``
    keeps up-to-date, of the form::

        {"query": ..., "rubrics": [{"title": ..., "description": ..., "weight": +/-1}, ...],
         "rubrics_types": ["persistent", ..., "evolving", ...]}

    Both the static (persistent) rubrics shipped with the dataset and any active evolving
    rubrics generated during training appear in ``rubrics``; this helper just renders them
    as a positive/negative criteria prefix so the value model is conditioned on the same
    criteria the verifier will use to grade the response.

    Token budget is enforced approximately (4 chars ~= 1 token) so a long rubric set never
    crowds out the rollout in the value forward.
    """
    if not ground_truth:
        return ""
    try:
        gt_obj = json.loads(ground_truth)
    except (TypeError, ValueError, json.JSONDecodeError):
        logger.debug("rubric conditioning: ground_truth was not valid JSON; skipping conditioning")
        return ""
    if not isinstance(gt_obj, dict):
        return ""
    rubrics = gt_obj.get("rubrics") or []
    if not isinstance(rubrics, list) or not rubrics:
        return ""

    positive_lines: list[str] = []
    negative_lines: list[str] = []
    budget = _RUBRIC_CONDITIONING_MAX_TOKENS
    for rubric in rubrics:
        if not isinstance(rubric, dict):
            continue
        title = str(rubric.get("title", "")).strip()
        description = str(rubric.get("description", "")).strip()
        if not description and not title:
            continue
        try:
            weight = float(rubric.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        body = f"{title}: {description}" if title and description else (title or description)
        line = f"- {body}\n"
        approx_tokens = max(1, len(line) // 4)
        if approx_tokens > budget:
            continue
        budget -= approx_tokens
        if weight >= 0:
            positive_lines.append(line)
        else:
            negative_lines.append(line)

    if not positive_lines and not negative_lines:
        return ""

    parts: list[str] = ["The final response will be graded against the following criteria.\n"]
    if positive_lines:
        parts.append("Positive criteria (the response should satisfy these):\n")
        parts.extend(positive_lines)
    if negative_lines:
        parts.append("Negative criteria (the response should NOT satisfy these):\n")
        parts.extend(negative_lines)
    parts.append("\n")
    return "".join(parts)


_SCORE_RE = re.compile(r"<answer>\s*([-+]?[0-9]*\.?[0-9]+)\s*</answer>")


def build_generative_value_prompt(
    partial_response: str,
    conditioning: str,  # one of: "none", "gt", "correct_demo", "rollout_context"
    ground_truth: str = "",
    siblings: Sequence[dict] | None = None,
    score_min: float = 0.0,
    score_max: float = 10.0,
    problem: str = "",
) -> str:
    """Build the gen-value prompt.

    Mirrors the template in Figure 3 of GenAC (arXiv:2604.10701): the critic sees the
    original ``problem`` followed by a ``partial_response`` and is asked to reason
    briefly before emitting ``<answer>X</answer>`` with X in [score_min, score_max].
    Generation stops on ``</answer>``; scores are later rescaled to [0, 1].
    """
    conditioning_text = ""
    if conditioning == "gt" and ground_truth:
        conditioning_text = f"The correct answer is {ground_truth}. "
    elif conditioning == "correct_demo" and siblings:
        chosen = next((s for s in siblings if s.get("is_correct")), siblings[0])
        tag = "CORRECT" if chosen.get("is_correct") else "INCORRECT"
        conditioning_text = f"Here is a reference attempt ({tag}):\n{str(chosen.get('text', ''))}\n"
    elif conditioning == "rollout_context" and siblings:
        lines = []
        budget = 4096
        for k, s in enumerate(siblings):
            tag = "CORRECT" if s.get("is_correct") else "INCORRECT"
            line = f"Attempt {k + 1} ({tag}):\n{str(s.get('text', ''))}\n"
            approx = max(1, len(line) // 4)
            if approx > budget:
                continue
            budget -= approx
            lines.append(line)
        conditioning_text = "Here are some other attempts at this question:\n" + "".join(lines)

    # Instruction template mirrors Figure 3 of GenAC (arXiv:2604.10701), minus the
    # In-Context Conditioning item (paper's step 2), which is intentionally omitted.
    instruction = (
        "You will be given a problem and a partial response. Your job is to predict the "
        f"expected value of the response on an integer scale from {int(score_min)} (very "
        f"unlikely to succeed) to {int(score_max)} ({int(score_max)} most likely).\n"
        "\n"
        "Instructions:\n"
        "1. Evaluate the difficulty of the problem.\n"
        "2. Skim through the partial solution and detect any progress, error, or confusion.\n"
        "3. Analyze the probability of success if the model finishes the solution.\n"
        f"4. Output your final answer as an integer between {int(score_min)} and "
        f"{int(score_max)} inclusive, wrapped in <answer>...</answer>."
    )
    problem_block = f"Problem:\n{problem}\n\n" if problem else ""
    conditioning_block = f"{conditioning_text}\n" if conditioning_text else ""
    return (
        f"{instruction}\n\n"
        f"{problem_block}"
        f"{conditioning_block}"
        f"Partial response:\n<rollout>{partial_response}</rollout>\nAnswer:"
    )


def parse_generative_value_score(text: str, score_min: float = 0.0, score_max: float = 10.0) -> float | None:
    """Extract the score from a `{score: X}` pattern."""
    m = _SCORE_RE.search(text)
    if m:
        try:
            v = float(m.group(1))
            return max(score_min, min(score_max, v))
        except ValueError:
            return None
    return None


def value_clipped_mse_loss(
    new_values: torch.Tensor,
    returns: torch.Tensor,
    old_values: torch.Tensor | None,
    mask: torch.Tensor,
    clip_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PPO2-style clipped value loss. Returns (per_token_loss, clipfrac)."""
    mask_f = mask.float()
    vf_losses1 = (new_values - returns).pow(2)
    if clip_range > 0 and old_values is not None:
        values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
        vf_losses2 = (values_clipped - returns).pow(2)
        per_token = torch.maximum(vf_losses1, vf_losses2)
        clipfrac = ((vf_losses2 > vf_losses1).float() * mask_f).sum() / mask_f.sum().clamp(min=1)
    else:
        per_token = vf_losses1
        clipfrac = torch.zeros((), dtype=torch.float32, device=new_values.device)
    return per_token * mask_f, clipfrac
