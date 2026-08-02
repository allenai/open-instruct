"""Turning several reward dimensions into the one number GRPO consumes.

Summing dimensions and then normalising - which is what any single scalar
reward does - lets the highest-variance dimension own the gradient no matter
what weights you intended. Normalising each dimension inside the group FIRST
makes every dimension contribute equally and makes the result invariant to each
one's scale, so there are no reward weights left to hand-tune:

    A_k(t_i) = (r_k(t_i) - mean_k) / (std_k + eps)
    A(t_i)   = (1 / |K|) * sum_k A_k(t_i)

This is MO-GRPO (arXiv:2509.22047), which proves the scale-invariance property
and validates it at 3B; PEARL's Eq. 11-12 is the same estimator, and TRL ships
it as ``multi_objective_aggregation="normalize_then_sum"``.

HOW THIS COMPOSES WITH open-instruct'S ADVANTAGE STEP. ``data_loader`` computes
``advantages = scores - group_mean`` (``advantage_normalization_type="centered"``,
the default). The output of ``normalize_then_sum`` already has zero mean within
the group, so that subtraction is a no-op and the estimator above survives
exactly. Under ``"standard"`` it is additionally divided by its own std, which
is a positive rescale and changes no sign. Either way you get what you asked
for - but leave it on "centered" if you want the arithmetic to be literally the
equation above.
"""

from __future__ import annotations

import math
import statistics

DimensionRow = dict[str, float | None]


def normalize_then_sum(rows: list[DimensionRow], names: tuple[str, ...] | list[str], eps: float = 1e-4) -> list[float]:
    """One scalar per group member, from per-dimension scores.

    A dimension with no spread across the group contributes exactly zero rather
    than exploding through the epsilon. If all eight completions scored the same
    on `brevity`, brevity has nothing to say about which of them is better, and
    the dimensions that do vary should decide alone.

    It is still counted in the denominator, because the estimator divides by a
    fixed |K|. So as dimensions saturate late in training the advantage shrinks
    and the effective learning rate falls with it. That is the published form
    and it is what TRL does; if you would rather it did not, divide by the
    number of dimensions that actually varied - but then a group where one
    dimension survives produces advantages as large as a group where six do,
    which is its own distortion.

    A ``None`` is treated as the dimension's group mean, which makes it
    contribute nothing to that member's advantage instead of dragging it toward
    an invented value. Prefer to drop such samples - ``count_missing`` is there
    so you can.
    """
    if not rows:
        return []
    n = len(rows)
    per_dimension: list[list[float]] = []
    for name in names:
        present = [float(r[name]) for r in rows if r.get(name) is not None]
        if not present:
            per_dimension.append([0.0] * n)
            continue
        mean = statistics.fmean(present)
        values = [float(r[name]) if r.get(name) is not None else mean for r in rows]
        sd = statistics.pstdev(values)
        if sd <= eps:
            per_dimension.append([0.0] * n)
        else:
            per_dimension.append([(v - mean) / (sd + eps) for v in values])
    if not per_dimension:
        return [0.0] * n
    return [statistics.fmean([d[i] for d in per_dimension]) for i in range(n)]


def centered(scores: list[float]) -> list[float]:
    """Subtract the group mean. The single-dimension case of the above."""
    if not scores:
        return []
    mean = statistics.fmean(scores)
    return [s - mean for s in scores]


def z_scored(scores: list[float], eps: float = 1e-4) -> list[float]:
    if not scores:
        return []
    mean = statistics.fmean(scores)
    sd = statistics.pstdev(scores)
    if sd <= eps:
        return [0.0] * len(scores)
    return [(s - mean) / (sd + eps) for s in scores]


def count_missing(rows: list[DimensionRow], names) -> int:
    """How many (member, dimension) cells the scorer could not fill."""
    return sum(1 for r in rows for name in names if r.get(name) is None)


def is_degenerate(values: list[float], eps: float = 1e-9) -> bool:
    """Does this group produce any gradient at all?

    A group whose members all score the same contributes exactly zero to the
    GRPO update, however large the scores are. Watch the fraction of groups in
    this state from step 0: when it approaches 1.0 the reward has stopped
    resolving differences and no amount of further training will help.
    """
    return all(abs(v - values[0]) < eps for v in values) if values else True


def zero_advantage_fraction(groups: list[list[float]]) -> float:
    if not groups:
        return math.nan
    return sum(1 for g in groups if is_degenerate(g)) / len(groups)
