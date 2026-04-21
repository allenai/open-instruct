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

import re
from collections.abc import Sequence

import torch

_ROLLOUT_CONTEXT_MAX_TOKENS = 4096
_POSTFIX_TEMPLATES = {
    "expected_accuracy",
    "rollout_context",
    "correct_demo",
}
_PREFIX_TEMPLATES = {"answer_prefix"}

# Templates that need a ground-truth string to be meaningful.
TEMPLATES_REQUIRING_GT: frozenset[str] = frozenset({"rollout_context", "correct_demo"})
# Templates that need sibling rollouts (decoded responses from the same prompt group).
TEMPLATES_REQUIRING_SIBLINGS: frozenset[str] = frozenset({"rollout_context", "correct_demo"})
# All valid gt_conditioning_template values.
ALL_GT_CONDITIONING_TEMPLATES: frozenset[str] = frozenset(_POSTFIX_TEMPLATES | _PREFIX_TEMPLATES)
# Valid values for --gen_value_conditioning.
GEN_VALUE_CONDITIONING_TYPES: frozenset[str] = frozenset({"none", "gt", "correct_demo", "rollout_context"})


def rescale_gen_value_score(parsed: float, score_min: float, score_max: float) -> float:
    """Rescale a raw gen-value score from [score_min, score_max] to [0, 1]."""
    return max(0.0, min(1.0, (parsed - score_min) / max(score_max - score_min, 1e-8)))


def is_postfix_template(template: str) -> bool:
    """Postfix templates are spliced BETWEEN prompt and response (per sub-sequence).

    Prefix templates are prepended to the entire packed sub-sequence.
    """
    return template in _POSTFIX_TEMPLATES


def build_conditioning_text(
    template: str,
    ground_truth: str,
    siblings: Sequence[dict] | None = None,
) -> str:
    """Return the conditioning text to splice for a single sub-sequence.

    The text is inserted between the prompt and the response (postfix templates) or as a prefix to
    the whole sub-sequence (prefix templates). Callers are expected to tokenize the returned
    string and extend position ids accordingly.
    """
    if template == "answer_prefix":
        return f"Note the correct answer is: {ground_truth}\n"
    if template == "expected_accuracy":
        return f"Given the answer is {ground_truth}, Let me compute the expected accuracy of the partial rollout: "
    if template == "rollout_context":
        return _build_rollout_context(ground_truth, siblings or [])
    if template == "correct_demo":
        return _build_correct_demo_context(ground_truth, siblings or [])
    raise ValueError(f"Unknown gt_conditioning_template: {template!r}")


def _build_rollout_context(ground_truth: str, siblings: Sequence[dict]) -> str:
    header = "Here are some other attempts at this question:\n"
    suffix = (
        f"Given the answer is {ground_truth}, compute the expected accuracy of the current attempt: "
    )
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
    return (
        reference
        + f"Given the answer is {ground_truth}, compute the expected accuracy of the current attempt: "
    )


_SCORE_RE = re.compile(r"\{\s*score\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*\}")
_LEADING_DIGIT_RE = re.compile(r"^\s*(-?[0-9]+(?:\.[0-9]+)?)")


def build_generative_value_prompt(
    partial_response: str,
    conditioning: str,  # one of: "none", "gt", "correct_demo", "rollout_context"
    ground_truth: str = "",
    siblings: Sequence[dict] | None = None,
    score_min: float = 0.0,
    score_max: float = 10.0,
) -> str:
    """Build the gen-value prompt.

    Format: "<conditioning_prefix>Here is a given partial response. Please predict the expected
    value of the response, scoring between 0 and 10. Reason briefly, then output exactly
    {score: X}. <rollout>{partial}</rollout>\nThus, the score is "
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

    instruction = (
        f"Here is a given partial response. Please predict the expected value of the response, "
        f"scoring between {int(score_min)} and {int(score_max)}. "
        f"Reason briefly, then output exactly {{score: X}} where X is a number in "
        f"[{int(score_min)}, {int(score_max)}]."
    )
    suffix = f"<rollout>{partial_response}</rollout>\nThus, the score is "
    return f"{conditioning_text}{instruction} {suffix}"


def parse_generative_value_score(
    text: str,
    score_min: float = 0.0,
    score_max: float = 10.0,
) -> float | None:
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
        per_token = 0.5 * torch.maximum(vf_losses1, vf_losses2)
        clipfrac = ((vf_losses2 > vf_losses1).float() * mask_f).sum() / mask_f.sum().clamp(min=1)
    else:
        per_token = 0.5 * vf_losses1
        clipfrac = torch.zeros((), dtype=torch.float32, device=new_values.device)
    return per_token * mask_f, clipfrac
