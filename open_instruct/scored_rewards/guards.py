"""Wrappers that constrain a score you do not fully trust.

A learned or judged reward is a model, and the policy will optimise the model
rather than the thing you meant. These are the three interventions that turned
out to be worth having, each a wrapper over any ``Scorer`` so they compose:

``Veto``      a cheap, rule-based check that OVERRIDES the score when it fires.
``Gate``      a precondition that zeroes the score when it is not met.
``Contrast``  pay only for the part of the score that a counterfactual does not
              also earn.
``MultiDimensional``  turn per-dimension scores into one scalar, normalised
              within the group.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable, Iterable
from typing import Any

from open_instruct.scored_rewards import aggregate
from open_instruct.scored_rewards.types import GroupScorer, Sample, Scorer, ScoreResult


class Veto(Scorer):
    """Override the inner score with a floor when a rule fires.

    Use for the behaviour you can detect with certainty and want to make
    strictly worse than doing nothing - not for quality. The rule should be a
    RULE, not a second model: the point of a veto is that the policy cannot
    negotiate with it.

    KEEP THE FLOOR MODEST. GRPO normalises advantages within the group, so a
    huge penalty mostly inflates the group's standard deviation and crushes the
    signal among the samples that did not trip it. A floor one unit below the
    worst honest score already says "strictly worse than failing honestly",
    which is the strongest statement the normalisation can carry.

    The override REPLACES the score rather than adjusting it, so a false
    positive does not add noise - it discards everything else you measured about
    that sample. Calibrate the rule's precision before turning it on, and log
    ``<name>_fired`` to watch its rate.
    """

    def __init__(self, inner: Scorer, rule: Callable[[Sample], bool], floor: float = -1.0, name: str = "veto"):
        self.inner = inner
        self.rule = rule
        self.floor = float(floor)
        self.name = name

    async def score(self, sample: Sample) -> ScoreResult:
        result = await self.inner.score(sample)
        fired = bool(self.rule(sample))
        result.info[f"{self.name}_fired"] = float(fired)
        if fired:
            result.score = self.floor
            result.dimensions = {k: self.floor for k in result.dimensions}
        return result


class Gate(Scorer):
    """Zero the score unless a precondition holds.

    Distinct from ``Veto``: a gate says "this sample earned nothing", a veto
    says "this sample was worse than nothing". Use a gate for an incomplete or
    malformed rollout, where the right statement is that no reward was earned
    rather than that a rule was broken.
    """

    def __init__(self, inner: Scorer, predicate: Callable[[Sample, ScoreResult], bool], name: str = "gate"):
        self.inner = inner
        self.predicate = predicate
        self.name = name

    async def score(self, sample: Sample) -> ScoreResult:
        result = await self.inner.score(sample)
        passed = bool(self.predicate(sample, result))
        result.info[f"{self.name}_passed"] = float(passed)
        if not passed:
            result.score = 0.0
            result.dimensions = {k: 0.0 for k in result.dimensions}
        return result


class Contrast(GroupScorer):
    """Pay only for what a counterfactual does not also earn.

        reward_i = score(completion_i | own item) - score(completion_i | a foreign item)

    The problem it solves: a completion can raise a score two ways, by doing the
    task and by being the sort of text that raises this score on anything. Both
    look identical to an absolute measurement. Re-scoring the same completion
    against a deliberately mismatched item isolates the second and subtracts it.

    DIRECTION MATTERS, and the two available directions measure opposite things.
    Hold the COMPLETION fixed and vary the item, and you learn whether this
    completion is generic - which is what this class does. Hold the ITEM fixed
    and vary the completion, and you learn how easily this item yields to
    anything, which is not a property of the completion at all; GRPO would then
    credit a member for a sibling's work.

    THE FOREIGN ITEM HAS TO COME FROM OUTSIDE THE GROUP. This is the thing that
    does not survive the move onto open-instruct unchanged. A GRPO group is G
    samples of ONE prompt, so every member of ``group`` carries the SAME item -
    swapping items around inside the group returns each member its own item and
    the whole term collapses to zero. Draw the foreign item from a pool
    instead: ``Contrast(scorer, ItemPool(items))``.

    IT MUST ALSO VARY WITHIN THE GROUP, or it does nothing for a different
    reason. Advantages are mean-centred over the G completions, so any term
    identical across them cancels exactly:

        A_i  is proportional to  (s_i - mean s) - (c_i - mean c)

    Only the deviation of the subtracted term survives. That is why the pool
    hands the WHOLE GROUP one foreign item rather than one per member: the item
    is then constant and the only thing varying in the subtracted term is the
    member's own completion, which is exactly what you want it to be measuring.
    """

    def __init__(self, inner: Scorer, counterfactual: Callable[[list[Sample]], Any], name: str = "contrast"):
        self.inner = inner
        self.counterfactual = counterfactual
        self.name = name

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        own = await asyncio.gather(*(self.inner.score(s) for s in group))
        foreign = self.counterfactual(group)
        others = [with_item(s, foreign) for s in group]
        off = await asyncio.gather(*(self.inner.score(o) for o in others))
        out = []
        for base, alt in zip(own, off):
            base.info[f"{self.name}_own"] = base.score
            base.info[f"{self.name}_off"] = alt.score
            base.info[self.name] = base.score - alt.score
            base.score = base.score - alt.score
            base.dimensions = {
                k: (None if v is None or alt.dimensions.get(k) is None else v - float(alt.dimensions[k]))
                for k, v in base.dimensions.items()
            }
            out.append(base)
        return out


def with_item(sample: Sample, item: dict) -> Sample:
    """The same completion, re-labelled against a different item."""
    return Sample(
        completion=sample.completion,
        prompt=sample.prompt,
        label=item,
        token_ids=sample.token_ids,
        rollout=sample.rollout,
        index=sample.index,
        group_size=sample.group_size,
    )


class ItemPool:
    """Hands a group one foreign item, chosen deterministically and never its own.

    Deterministic on the group's own item, so two runs with the same seed draw
    the same comparison and the metric is reproducible. Constant across the
    group, for the mean-centring reason in ``Contrast``.
    """

    def __init__(self, items: list[dict], key: str = "question", seed: int = 0):
        self.items = list(items)
        self.key = key
        self.seed = seed

    def __call__(self, group: list[Sample]) -> dict:
        if not self.items:
            return {}
        own = group[0].item if group else {}
        digest = hashlib.blake2b(f"{own.get(self.key, '')}|{self.seed}".encode(), digest_size=8).digest()
        start = int.from_bytes(digest, "big") % len(self.items)
        for offset in range(len(self.items)):
            candidate = self.items[(start + offset) % len(self.items)]
            if candidate.get(self.key) != own.get(self.key):
                return candidate
        return self.items[start]


class MultiDimensional(GroupScorer):
    """Per-dimension scores in, one normalised scalar out.

    Wraps a scorer that fills ``ScoreResult.dimensions``. Each dimension is
    z-scored across the group and the member's reward is the mean of those,
    which is ``aggregate.normalize_then_sum``. Read that module for why the
    order matters and how it composes with open-instruct's own centring.
    """

    def __init__(self, inner: Scorer | GroupScorer, dimensions: Iterable[str], name: str = "multi_dimensional"):
        self.inner = inner
        self.dimensions = tuple(dimensions)
        self.name = name

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        if isinstance(self.inner, GroupScorer):
            results = await self.inner.score_group(group)
        else:
            results = list(await asyncio.gather(*(self.inner.score(s) for s in group)))
        rows = [{k: r.dimensions.get(k) for k in self.dimensions} for r in results]
        scalars = aggregate.normalize_then_sum(rows, self.dimensions)
        missing = aggregate.count_missing(rows, self.dimensions)
        for result, scalar in zip(results, scalars):
            result.info["dimensions_missing"] = float(missing)
            result.score = scalar
        return results


class Weighted(GroupScorer):
    """Sum several group scorers with fixed weights.

    Use when the terms are genuinely on the same scale and you mean the weights.
    When they are not, prefer ``MultiDimensional``, which removes the scales and
    therefore the need for weights.
    """

    def __init__(self, parts: dict[str, tuple[GroupScorer, float]], name: str = "weighted"):
        self.parts = parts
        self.name = name

    async def score_group(self, group: list[Sample]) -> list[ScoreResult]:
        names = list(self.parts)
        all_results = await asyncio.gather(*(self.parts[n][0].score_group(group) for n in names))
        out = [ScoreResult() for _ in group]
        for name, results in zip(names, all_results):
            weight = self.parts[name][1]
            for combined, part in zip(out, results):
                combined.score += weight * part.score
                combined.info[f"{name}_score"] = part.score
                combined.info.update({f"{name}_{k}": v for k, v in part.info.items()})
        return out
