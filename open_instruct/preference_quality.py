# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities for filtering RLAIF / synthetic preference data by quality signals.

These helpers sit *after* LLM-as-judge annotation (see
``open_instruct/rejection_sampling/synthetic_preference_dataset.py``) and *before*
reward modeling or DPO. They automate common preference-quality steps:

* drop identical chosen/rejected pairs
* drop low score-margin pairs when numeric judge/RM scores are present
* drop multi-judge disagreements (optional majority vote)
* report simple length-bias / retention stats

The goal is a small, testable building block for disagreement-aware or
curriculum-style preference filtering experiments.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def should_swap_after_shuffle(preferred: str, shuffled_index: int) -> bool:
    """Return whether ``chosen``/``rejected`` must be swapped after positional shuffle.

    When building a judge prompt we show ``comparison_pair[shuffled_index]`` as
    response 0 and ``comparison_pair[1 - shuffled_index]`` as response 1, but we
    always initialize ``chosen = comparison_pair[0]`` and
    ``rejected = comparison_pair[1]``. A swap is required exactly when the judge's
    preferred *display* index maps to ``comparison_pair[1]``.

    Args:
        preferred: Judge label, typically ``\"0\"`` or ``\"1\"`` (whitespace ignored).
        shuffled_index: Index of ``comparison_pair`` shown as response 0 (0 or 1).

    Returns:
        ``True`` if chosen/rejected should be swapped to match the judge.
    """
    preferred_norm = preferred.strip()
    if preferred_norm not in {"0", "1"}:
        raise ValueError(f"preferred must be '0' or '1', got {preferred!r}")
    if shuffled_index not in {0, 1}:
        raise ValueError(f"shuffled_index must be 0 or 1, got {shuffled_index!r}")
    # preferred "0" with shuffle 1 -> judge picked original pair[1]
    # preferred "1" with shuffle 0 -> judge picked original pair[1]
    return (preferred_norm == "0" and shuffled_index == 1) or (
        preferred_norm == "1" and shuffled_index == 0
    )


def _last_assistant_text(messages: Any) -> str | None:
    """Extract the last assistant message text from a chat-style preference arm."""
    if isinstance(messages, str):
        return messages
    if not isinstance(messages, Sequence) or isinstance(messages, (bytes, bytearray)):
        return None
    for message in reversed(list(messages)):
        if isinstance(message, Mapping) and message.get("role") == "assistant":
            content = message.get("content")
            return content if isinstance(content, str) else str(content)
        if isinstance(message, Mapping) and "content" in message and "role" not in message:
            # Some datasets store only content dicts; fall through to string form.
            content = message.get("content")
            return content if isinstance(content, str) else str(content)
    if messages and isinstance(messages[-1], Mapping) and "content" in messages[-1]:
        content = messages[-1]["content"]
        return content if isinstance(content, str) else str(content)
    return None


def is_identical_preference(
    row: Mapping[str, Any],
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
) -> bool:
    """Return True when chosen and rejected assistant texts are identical."""
    chosen = _last_assistant_text(row.get(chosen_key))
    rejected = _last_assistant_text(row.get(rejected_key))
    if chosen is None or rejected is None:
        return False
    return chosen.strip() == rejected.strip()


def score_margin(row: Mapping[str, Any], chosen_score_key: str, rejected_score_key: str) -> float | None:
    """Absolute difference between chosen and rejected numeric scores, if present."""
    try:
        chosen_score = float(row[chosen_score_key])
        rejected_score = float(row[rejected_score_key])
    except (KeyError, TypeError, ValueError):
        return None
    return abs(chosen_score - rejected_score)


def judges_agree(labels: Sequence[Any]) -> bool:
    """Return True when every non-null judge label equals the first non-null label."""
    normalized = [str(label).strip() for label in labels if label is not None and str(label).strip() != ""]
    if len(normalized) < 2:
        return True
    return all(label == normalized[0] for label in normalized[1:])


@dataclass
class PreferenceFilterStats:
    """Retention / bias summary for a preference filtering pass."""

    input_rows: int = 0
    kept_rows: int = 0
    dropped_identical: int = 0
    dropped_low_margin: int = 0
    dropped_disagreement: int = 0
    mean_chosen_len: float | None = None
    mean_rejected_len: float | None = None
    length_bias_ratio: float | None = None  # mean(chosen_len) / mean(rejected_len)

    @property
    def retention_rate(self) -> float:
        if self.input_rows == 0:
            return 0.0
        return self.kept_rows / self.input_rows

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["retention_rate"] = self.retention_rate
        return payload


def filter_preference_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    chosen_key: str = "chosen",
    rejected_key: str = "rejected",
    drop_identical: bool = True,
    min_score_margin: float | None = None,
    chosen_score_key: str = "chosen_score",
    rejected_score_key: str = "rejected_score",
    judge_label_key: str | None = None,
    require_judge_agreement: bool = False,
) -> tuple[list[dict[str, Any]], PreferenceFilterStats]:
    """Filter preference rows and collect simple quality statistics.

    Args:
        rows: Preference examples (dicts).
        drop_identical: Drop pairs whose final assistant texts match.
        min_score_margin: If set, drop pairs whose |chosen_score - rejected_score|
            is strictly below this threshold (rows missing scores are kept).
        judge_label_key: Optional key whose value is a list of judge labels.
        require_judge_agreement: If True, drop rows where judge labels disagree.

    Returns:
        ``(kept_rows, stats)``
    """
    kept: list[dict[str, Any]] = []
    stats = PreferenceFilterStats()
    chosen_lengths: list[int] = []
    rejected_lengths: list[int] = []

    for row in rows:
        stats.input_rows += 1
        row_dict = dict(row)

        if drop_identical and is_identical_preference(row_dict, chosen_key, rejected_key):
            stats.dropped_identical += 1
            continue

        if min_score_margin is not None:
            margin = score_margin(row_dict, chosen_score_key, rejected_score_key)
            if margin is not None and margin < min_score_margin:
                stats.dropped_low_margin += 1
                continue

        if require_judge_agreement and judge_label_key is not None:
            labels = row_dict.get(judge_label_key, [])
            if isinstance(labels, Sequence) and not judges_agree(labels):
                stats.dropped_disagreement += 1
                continue

        chosen_text = _last_assistant_text(row_dict.get(chosen_key))
        rejected_text = _last_assistant_text(row_dict.get(rejected_key))
        if chosen_text is not None:
            chosen_lengths.append(len(chosen_text))
        if rejected_text is not None:
            rejected_lengths.append(len(rejected_text))

        kept.append(row_dict)
        stats.kept_rows += 1

    if chosen_lengths:
        stats.mean_chosen_len = sum(chosen_lengths) / len(chosen_lengths)
    if rejected_lengths:
        stats.mean_rejected_len = sum(rejected_lengths) / len(rejected_lengths)
    if stats.mean_chosen_len is not None and stats.mean_rejected_len not in (None, 0):
        stats.length_bias_ratio = stats.mean_chosen_len / stats.mean_rejected_len

    return kept, stats


def filter_preference_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    stats_path: str | Path | None = None,
    **filter_kwargs: Any,
) -> PreferenceFilterStats:
    """Read a JSONL preference file, filter it, and write kept rows + optional stats."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    rows: list[dict[str, Any]] = []
    with input_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    kept, stats = filter_preference_rows(rows, **filter_kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    if stats_path is not None:
        stats_path = Path(stats_path)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n")

    return stats
