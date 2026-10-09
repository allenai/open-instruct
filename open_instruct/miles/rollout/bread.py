"""Search expert prefixes for learnable continuation groups before reward filtering."""

import copy
import dataclasses
import hashlib
import math
from collections.abc import Awaitable, Callable
from typing import Any


@dataclasses.dataclass(frozen=True)
class SearchResult:
    group: list[Any]
    event: dict


def rewards(group, reward_key):
    """Require finished binary-reward samples before treating a group as searchable."""
    values = []
    for sample in group:
        if getattr(sample.status, "value", sample.status) != "completed":
            return None
        value = sample.reward.get(reward_key) if isinstance(sample.reward, dict) else sample.reward
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value not in (0, 1):
            return None
        values.append(float(value))
    return values if len(values) >= 2 else None


def prefixes(chunks, max_prefixes):
    """Yield cumulative section boundaries without supplying the complete solution."""
    if not isinstance(chunks, list) or not all(isinstance(part, str) and part.strip() for part in chunks):
        raise ValueError("BREAD requires nonempty reference sections")
    for count in range(1, min(len(chunks), max_prefixes + 1)):
        prefix = "\n\n".join(chunks[:count]) + "\n\n"
        if "\\boxed" in prefix or "<answer>" in prefix:
            break
        yield count, prefix


def branch(pristine, prefix, count):
    """Append an assistant prefill to pristine prompts, outside the generated response."""
    result = copy.deepcopy(pristine)
    for sample in result:
        if not isinstance(sample.prompt, str) or sample.tokens or sample.response_length or sample.response:
            raise ValueError("BREAD requires fresh samples with rendered text prompts")
        sample.prompt += prefix
        sample.metadata = {**(sample.metadata or {}), "bread_anchor_sections": count}
    return result


async def rescue(
    pristine: list[Any],
    initial: list[Any],
    generate: Callable[[list[Any]], Awaitable[list[Any]]],
    *,
    reward_key="score",
    max_prefixes=9,
    fits: Callable[[str], bool] = lambda _: True,
) -> SearchResult:
    """Try progressively longer prefixes, then resample a mixed-success candidate."""
    event = {"initial_rewards": rewards(initial, reward_key), "attempts": [], "outcome": "unchanged"}
    if event["initial_rewards"] is None or any(event["initial_rewards"]):
        return SearchResult(initial, event)
    specs = [(sample.metadata or {}).get("bread") for sample in pristine]
    if not specs or specs[0] is None:
        return SearchResult(initial, event | {"outcome": "no_reference"})
    if any(spec != specs[0] for spec in specs):
        raise ValueError("BREAD group members must share the same reference")
    event["reference_sections"] = len(specs[0]["sections"])
    event["outcome"] = "exhausted"
    for count, prefix in prefixes(specs[0]["sections"], max_prefixes):
        group = branch(pristine, prefix, count)
        if not all(fits(sample.prompt) for sample in group):
            event["outcome"] = "context_limit"
            break
        probe = await generate(group)
        scores = rewards(probe, reward_key)
        event["attempts"].append(
            {
                "sections": count,
                "prefix_chars": len(prefix),
                "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
                "rewards": scores,
                "generated_tokens": sum(sample.response_length for sample in probe),
            }
        )
        if scores is None:
            event["outcome"] = "unfinished_or_nonbinary"
            return SearchResult(probe, event)
        if not any(scores):
            continue
        # A fresh group avoids using selection noise as the training outcome.
        fresh = await generate(branch(pristine, prefix, count))
        fresh_scores = rewards(fresh, reward_key)
        event.update(
            selected_sections=count,
            fresh_rewards=fresh_scores,
            fresh_generated_tokens=sum(sample.response_length for sample in fresh),
        )
        if fresh_scores is not None and 0 < sum(fresh_scores) < len(fresh_scores):
            event["outcome"] = "rescued"
        else:
            event["outcome"] = "fresh_group_not_mixed"
        return SearchResult(fresh, event)
    return SearchResult(initial, event)
