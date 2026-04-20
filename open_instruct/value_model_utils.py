# Copyright 2026 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Helpers for the PPO / LM-yesno / SAE value model used by grpo_fast.py.

The value model itself is built, optimized, and DeepSpeed-managed inside
`PolicyTrainerRayProcess.from_pretrained`; this module provides stateless helpers for:

- building value-conditioning strings from ground truths + sibling rollouts;
- running the value forward with or without between-prompt-and-response conditioning;
- extracting per-token values for scalar and LM-yesno heads;
- the binary cross-entropy loss used by the LM-yesno value model.
"""
from __future__ import annotations

import re
from typing import Sequence

import torch
import torch.nn.functional as F

_ROLLOUT_CONTEXT_MAX_TOKENS = 4096
_POSTFIX_TEMPLATES = {
    "expected_accuracy",
    "rollout_context",
    "correct_demo",
    "lm_yesno",
    "lm_yesno_blind",
    "lm_yesno_siblings",
}
_PREFIX_TEMPLATES = {"answer_prefix", "boxed_answer", "cot_spoiler"}


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
        return f"Answer: {ground_truth}\n"
    if template == "boxed_answer":
        return f"The correct answer is \\boxed{{{ground_truth}}}.\n"
    if template == "cot_spoiler":
        return (
            f"Therefore, the final answer is \\boxed{{{ground_truth}}}.\n"
            "Now let me show my working for this problem:\n"
        )
    if template == "expected_accuracy":
        return f"Given the answer is {ground_truth}, Let me compute the expected accuracy of the partial rollout: "
    if template == "lm_yesno":
        return (
            f"Answer: {ground_truth}. Here is a partial attempt. "
            "Will this attempt get the answer? Yes/no.\n"
        )
    if template == "lm_yesno_blind":
        return "Here is a partial attempt. Will this attempt get the answer? Yes/no.\n"
    if template == "rollout_context":
        return _build_rollout_context(ground_truth, siblings or [])
    if template == "correct_demo":
        return _build_correct_demo_context(ground_truth, siblings or [])
    if template == "lm_yesno_siblings":
        return _build_lm_yesno_siblings_context(ground_truth, siblings or [])
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


def _build_lm_yesno_siblings_context(ground_truth: str, siblings: Sequence[dict]) -> str:
    header = f"Answer: {ground_truth}\n"
    lines: list[str] = []
    for k, s in enumerate(siblings):
        tag = "success" if s.get("is_correct") else "fail"
        text = str(s.get("text", ""))
        lines.append(f"Attempt {k + 1}: {text} Result: {tag}\n")
    suffix = (
        "Here is a partial attempt. Will this attempt get the answer? Yes/no.\n"
        f"Attempt {len(siblings) + 1}: "
    )
    return header + "".join(lines) + suffix


def extract_yes_no_value(logits: torch.Tensor, yes_id: int, no_id: int) -> torch.Tensor:
    """Return softmax probability of the Yes token, restricted to {yes, no} logits.

    Args:
        logits: (..., vocab)
        yes_id: token id for "Yes"/"yes"
        no_id: token id for "No"/"no"
    Returns: (...) tensor in [0, 1].
    """
    yes_logit = logits[..., yes_id]
    no_logit = logits[..., no_id]
    pair = torch.stack([yes_logit, no_logit], dim=-1)
    return F.softmax(pair, dim=-1)[..., 0]


def lm_yesno_bce_loss(
    predicted_prob_yes: torch.Tensor,
    target_is_correct: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token BCE loss. All tensors are the same shape; `mask` is float/bool."""
    eps = 1e-6
    p = predicted_prob_yes.clamp(eps, 1.0 - eps)
    t = target_is_correct.float()
    loss = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    return loss * mask.float()


_SCORE_RE = re.compile(r"\{\s*score\s*:\s*([-+]?[0-9]*\.?[0-9]+)\s*\}")
_LEADING_DIGIT_RE = re.compile(r"^\s*(-?[0-9]+(?:\.[0-9]+)?)")


def build_generative_value_prompt(
    partial_response: str,
    conditioning: str,  # one of: "none", "gt", "correct_demo", "rollout_context"
    ground_truth: str = "",
    siblings: Sequence[dict] | None = None,
    allow_cot: bool = False,
    score_min: float = 0.0,
    score_max: float = 10.0,
) -> str:
    """Build the gen-value prompt described in the plan.

    Default form (no CoT, score 0..score_max):
        "<conditioning_prefix>Here is a given partial response. Please predict the expected
         value of the response, scoring between 0 and 10. <rollout>{partial}</rollout>
         Thus, the score is: "

    With `allow_cot=True`, switches to a `{score: X}` suffix.
    """
    conditioning_text = ""
    if conditioning == "gt" and ground_truth:
        conditioning_text = f"The correct answer is {ground_truth}. "
    elif conditioning == "correct_demo" and siblings:
        chosen = next((s for s in siblings if s.get("is_correct")), siblings[0])
        tag = "CORRECT" if chosen.get("is_correct") else "INCORRECT"
        conditioning_text = (
            f"Here is a reference attempt ({tag}):\n{str(chosen.get('text', ''))}\n"
        )
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

    if allow_cot:
        instruction = (
            f"Here is a given partial response. Please predict the expected value of the response, "
            f"scoring between {int(score_min)} and {int(score_max)}. "
            f"Reason briefly, then output exactly {{score: X}} where X is a number in "
            f"[{int(score_min)}, {int(score_max)}]."
        )
        suffix = (
            f"<rollout>{partial_response}</rollout>\nThus, the score is "
        )
    else:
        instruction = (
            f"Here is a given partial response. Please predict the expected value of the response, "
            f"scoring between {int(score_min)} and {int(score_max)}."
        )
        suffix = (
            f"<rollout>{partial_response}</rollout> Thus, the score is: "
        )
    return f"{conditioning_text}{instruction} {suffix}"


def parse_generative_value_score(
    text: str,
    score_min: float = 0.0,
    score_max: float = 10.0,
    allow_cot: bool = False,
) -> float | None:
    """Extract the score. With CoT, look for `{score: X}`; otherwise take the leading number."""
    if allow_cot:
        m = _SCORE_RE.search(text)
        if m:
            try:
                v = float(m.group(1))
                return max(score_min, min(score_max, v))
            except ValueError:
                return None
        return None
    m = _LEADING_DIGIT_RE.search(text)
    if not m:
        return None
    try:
        v = float(m.group(1))
        return max(score_min, min(score_max, v))
    except ValueError:
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
