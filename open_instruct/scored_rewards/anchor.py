"""A metric that is not the reward.

The failure mode that a judged or learned reward makes easy: the reward climbs,
every dashboard is green, and the policy has learned the judge rather than the
task. It is not detectable from inside - a judge cannot report that it is being
gamed, and evaluating a judge-shaped policy with the judge that shaped it will
confirm whatever the training curve said.

The only defence is a number computed by something that is not in the reward, on
items the policy does not train on. That is an anchor. It should be boring,
mechanical, and unchanged across every run you want to compare, because its
whole job is to be able to say "this did not work" four runs in a row.

WHAT IT MEASURES. Three conditions on the same held-out items:

    baseline   the outcome with no policy output at all
    treated    the outcome given THIS item's policy output
    swapped    the outcome given ANOTHER item's policy output

    gain         = treated - baseline
    specificity  = treated - swapped

``gain`` alone cannot tell teaching from filler. Plenty of text raises an
outcome on ANY item - "read each option carefully and rule out the silly ones" -
and it costs the policy no understanding at all. ``specificity`` separates them:
output written for this item stops working on a different one, and filler does
not. A gain that is entirely non-specific is a gain you should not celebrate.

KEEP THE CONDITIONS IDENTICAL IN EVERYTHING BUT THE THING VARIED. In one run an
oracle early-stop lived inside the loop that produced ``treated`` but not
``swapped``, so the treated condition got several attempts and the swapped
condition got one. The resulting specificity was inflated and not comparable
with the runs on either side of it. If the two conditions differ in anything
except which item the output was written for, the difference you measure is
that.

HOW BIG DOES THE SET NEED TO BE? For a binary outcome, one standard error is
about ``0.5 / sqrt(n)`` and the smallest difference you can trust is roughly
twice that:

    n=40   SE 0.079   min detectable ~0.16
    n=150  SE 0.041   min detectable ~0.082
    n=200  SE 0.035   min detectable ~0.071
    n=400  SE 0.025   min detectable ~0.050

At n=40 the smallest trustworthy difference is larger than most effects worth
chasing, so per-eval movement is noise. Budget ~200 items.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import statistics
from collections.abc import Awaitable, Callable, Sequence

#: ``(item, policy_text) -> outcome in [0, 1]``. Pass an empty string for the
#: baseline condition. This is the thing that must never appear in the reward.
OutcomeFn = Callable[[dict, str], Awaitable[float]]
#: ``(item) -> the policy's output for it``.
PolicyFn = Callable[[Sequence[dict]], Awaitable[list[str]]]


@dataclasses.dataclass
class AnchorResult:
    n: int
    baseline: float
    treated: float
    swapped: float
    extras: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def gain(self) -> float:
        return self.treated - self.baseline

    @property
    def specificity(self) -> float:
        return self.treated - self.swapped

    @property
    def standard_error(self) -> float:
        """SE of a proportion at this n. The scale below which nothing is real."""
        return math.sqrt(0.25 / self.n) if self.n else math.nan

    def to_dict(self) -> dict[str, float]:
        return {
            "anchor/n": float(self.n),
            "anchor/baseline": self.baseline,
            "anchor/treated": self.treated,
            "anchor/swapped": self.swapped,
            "anchor/gain": self.gain,
            "anchor/specificity": self.specificity,
            "anchor/se": self.standard_error,
            **{f"anchor/{k}": v for k, v in self.extras.items()},
        }

    def __str__(self) -> str:
        return (
            f"n={self.n}  baseline={self.baseline:.3f}  treated={self.treated:.3f}  "
            f"swapped={self.swapped:.3f}  gain={self.gain:+.3f}  "
            f"specificity={self.specificity:+.3f}  (1 SE = {self.standard_error:.3f})"
        )


class Anchor:
    """Held-out gain and specificity, computed outside the reward.

    ``extra_metrics`` lets you carry through a rule you also want reported
    honestly - typically the same veto rule the reward uses, so you can see
    whether a rising outcome is the policy getting better or the policy simply
    breaking the rule less often. Report both: an outcome that improves only
    because a penalised behaviour stopped is not the same result as an outcome
    that improves on the samples that never triggered it.
    """

    def __init__(
        self,
        items: Sequence[dict],
        policy: PolicyFn,
        outcome: OutcomeFn,
        extra_metrics: dict[str, Callable[[dict, str], float]] | None = None,
        concurrency: int = 16,
    ):
        self.items = list(items)
        self.policy = policy
        self.outcome = outcome
        self.extra_metrics = extra_metrics or {}
        self.concurrency = concurrency

    async def run(self) -> AnchorResult:
        if not self.items:
            return AnchorResult(0, math.nan, math.nan, math.nan)

        outputs = await self.policy(self.items)
        if len(outputs) != len(self.items):
            raise ValueError(f"policy returned {len(outputs)} outputs for {len(self.items)} items")
        # rotate by one: every item is scored against exactly one foreign output,
        # and every output is used exactly once in each condition
        swapped = outputs[1:] + outputs[:1]

        limiter = asyncio.Semaphore(self.concurrency)

        async def measure(item: dict, text: str) -> float:
            async with limiter:
                return float(await self.outcome(item, text))

        baseline, treated, swap = await asyncio.gather(
            asyncio.gather(*(measure(i, "") for i in self.items)),
            asyncio.gather(*(measure(i, t) for i, t in zip(self.items, outputs))),
            asyncio.gather(*(measure(i, s) for i, s in zip(self.items, swapped))),
        )

        extras: dict[str, float] = {}
        for name, fn in self.extra_metrics.items():
            values = [float(fn(i, t)) for i, t in zip(self.items, outputs)]
            extras[name] = statistics.fmean(values)
            # the outcome restricted to samples the rule did NOT flag; a rising
            # headline number with a flat clean number is the rule, not the task
            clean = [t for t, v in zip(treated, values) if v == 0.0]
            extras[f"clean_{name}"] = statistics.fmean(clean) if clean else math.nan

        return AnchorResult(
            n=len(self.items),
            baseline=statistics.fmean(baseline),
            treated=statistics.fmean(treated),
            swapped=statistics.fmean(swap),
            extras=extras,
        )


def moved(before: AnchorResult, after: AnchorResult, key: str = "treated") -> str:
    """A one-line verdict, in units of the standard error.

    Print this rather than the two numbers. A change of half a standard error is
    not a change, and stating it in SE makes that impossible to misread.
    """
    delta = getattr(after, key) - getattr(before, key)
    se = after.standard_error
    ratio = delta / se if se and not math.isnan(se) else math.nan
    verdict = "moved" if abs(ratio) >= 2 else "did not move"
    return f"anchor/{key} {delta:+.3f} = {ratio:+.1f} SE -> {verdict}"
