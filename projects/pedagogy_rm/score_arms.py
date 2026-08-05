"""Unblind the rated pool and ask whether the trained arms are actually better.

    python projects/pedagogy_rm/score_arms.py \
        --key data/eval50/key.json \
        --labels 'data/eval50/labels/agent_*.json' \
        --human data/eval50/labels/sophia.json

THE QUESTION THIS ANSWERS, which is the whole point of the run. GRPO optimised the probe,
and the probe's reward went up. That is not evidence of anything on its own: a policy that
learns to satisfy a Ridge head has learned to satisfy a Ridge head. The claim worth making
is that the turns got better as judged by something that never saw the reward, and the only
such judges here are the human and the agents calibrated to the human.

So the table prints both, per arm, side by side: what the probe thought, and what the raters
thought. The interesting cell is a dimension where those two disagree - a probe score that
rose while the ratings did not is the signature of reward hacking, and it is invisible if
you only ever look at one of the two columns.

PAIRED BY MOMENT, BECAUSE THE ALTERNATIVE THROWS AWAY MOST OF THE POWER. Every arm answered
the same prompts, so the comparison is within a prompt: the difficulty of the question and
the state of the student are held fixed, and between them they are the largest source of
variance in an absolute rating. Comparing arm means unpaired would drown a real half-point
effect in question difficulty. The confidence interval is over moments for the same reason.

WHAT A DIFFERENCE HERE IS AND IS NOT. The raters are calibrated to one person and validated
on 25 turns she labelled and they never saw; whatever correlation they achieved there is the
ceiling on how much any of this means. A dimension the raters could not reproduce - measured
by agreement.py, not assumed - carries no weight no matter how large its difference looks,
and is marked rather than dropped, because a difference on an unreliable dimension is
evidence about the dimension.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import math
import statistics

from projects.pedagogy_rm.rubric import BY_KEY, DIMENSIONS

# Which way is good, per dimension. Not derivable from the rubric objects, which only know
# the range, and not importable from plugin.py, which would drag torch in for four floats -
# so it is stated here and must stay consistent with that file's SIGNS.
#
# "mid" is the one that matters. length_fit scores 2 for the right length and 1 and 3 for
# too short and too long, so its mean is not a quality at all: an arm whose turns are half
# too short and half too long averages exactly 2 and looks perfect. Reporting the mean of a
# non-monotonic dimension is not a rounding error, it is the wrong measurement, and it would
# have hidden precisely the failure this eval was built to look for.
BETTER = {"leak": "low", "targeted": "high", "actionable": "high", "elicits": "high",
          "length_fit": "mid", "correct": "high"}


def goodness(key: str, value: float) -> float:
    """One rating re-expressed so that larger is always better.

    For "mid" dimensions this is the negated distance from the ideal, which turns
    "how long is it" into "how wrong is the length" and makes the arms comparable.
    """
    how = BETTER.get(key, "high")
    if how == "low":
        return -value
    if how == "mid":
        return -abs(value - 2.0)
    return value


def load_raters(patterns: list[str]) -> dict[str, dict[str, dict]]:
    """``{rater: {unit_id: record}}`` over every file the patterns match."""
    out: dict[str, dict[str, dict]] = {}
    for path in sorted({p for pattern in patterns for p in (glob.glob(pattern) or [pattern])}):
        with open(path) as handle:
            blob = json.load(handle)
        rater = blob.get("rater") or path.rsplit("/", 1)[-1].removesuffix(".json")
        out[rater] = {r["id"]: r for r in blob.get("labels", [])}
    return out


def consensus(raters: dict[str, dict[str, dict]], unit_id: str, key: str) -> float | None:
    """The mean rating for one turn on one dimension, or None if nobody scored it."""
    votes = [r[unit_id][key] for r in raters.values() if isinstance(r.get(unit_id, {}).get(key), int)]
    return statistics.fmean(votes) if votes else None


def paired(by_moment: dict[str, dict[str, float]], one: str, two: str) -> tuple[int, float, float, float]:
    """n, mean difference, half-width of its 95% interval, and the win rate for `one`."""
    diffs = [m[one] - m[two] for m in by_moment.values() if one in m and two in m]
    if len(diffs) < 3:
        return len(diffs), float("nan"), float("nan"), float("nan")
    mean = statistics.fmean(diffs)
    # 1.96 rather than a t quantile: at sixty moments the difference is under 2% and this
    # keeps the module free of scipy, which the cluster environment does not carry.
    half = 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))
    wins = statistics.fmean([1.0 if d > 0 else 0.0 if d < 0 else 0.5 for d in diffs])
    return len(diffs), mean, half, wins


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key", default="data/eval50/key.json")
    parser.add_argument("--labels", nargs="+", default=["data/eval50/labels/agent_*.json"])
    parser.add_argument("--human", default="", help="scored separately as well, on whatever it covers")
    parser.add_argument("--dimensions", default="", help="comma-separated keys; default is DIMENSIONS")
    parser.add_argument("--baseline", default="base", help="the arm the others are compared against")
    parser.add_argument(
        "--total",
        default="leak,targeted,actionable,elicits",
        help="dimensions scalarised into a reward-comparable total; must match the head's",
    )
    args = parser.parse_args()

    dims = DIMENSIONS if not args.dimensions else tuple(BY_KEY[k] for k in args.dimensions.split(","))
    with open(args.key) as handle:
        key = json.load(handle)["key"]
    arms = sorted({v["arm"] for v in key.values()})

    panels = [("agents", load_raters(args.labels))]
    if args.human:
        panels.append(("human", load_raters([args.human])))

    totals: dict[str, dict[str, float]] = {}
    for panel, raters in panels:
        if not raters or not any(raters.values()):
            continue
        covered = {u for r in raters.values() for u in r}
        moments = {key[u]["moment"] for u in covered if u in key}
        print(f"\n{'=' * 78}\n{panel.upper()}: {len(raters)} rater(s), {len(covered)} turns, {len(moments)} moments")
        print(f"  {', '.join(sorted(raters))}")

        # The same scalarisation the head applies, over the same dimensions, so this number
        # and the probe's `total` are the same quantity measured two ways: what the reward
        # said a turn was worth, and what the raters would have paid for it. Reporting only
        # per-dimension differences leaves the reader to combine them by eye, and the whole
        # question is whether the combination the policy optimised tracks the one people mean.
        wanted = [k for k in args.total.split(",") if k]
        scalar: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for unit_id in covered:
            entry = key.get(unit_id)
            parts = [goodness(k, v) for k in wanted if (v := consensus(raters, unit_id, k)) is not None]
            if entry and len(parts) == len(wanted):
                scalar[entry["moment"]][entry["arm"]] = statistics.fmean(parts)
        if scalar:
            totals[panel] = {
                arm: statistics.fmean([m[arm] for m in scalar.values() if arm in m])
                for arm in arms
                if any(arm in m for m in scalar.values())
            }
            print("\n  TOTAL       " + "  ".join(f"{a} {totals[panel][a]:.2f}" for a in arms if a in totals[panel]))
            for arm in arms:
                if arm == args.baseline or arm not in totals[panel]:
                    continue
                n, mean, half, wins = paired(scalar, arm, args.baseline)
                if n >= 3:
                    verdict = "  --  " if abs(mean) <= half else ("BETTER" if mean > 0 else " WORSE")
                    print(
                        f"    {arm:>8} vs {args.baseline:<8} {mean:+.2f} +/- {half:.2f}  "
                        f"wins {wins:.0%} of {n} moments  {verdict}"
                    )

        for dim in dims:
            # Turn scores first, then one number per (moment, arm), so an arm that happens
            # to have been rated by more people at some moment does not get extra weight.
            per_moment: dict[str, dict[str, float]] = collections.defaultdict(dict)
            raw: dict[str, list[float]] = collections.defaultdict(list)
            for unit_id in covered:
                entry = key.get(unit_id)
                value = consensus(raters, unit_id, dim.key)
                if entry and value is not None:
                    per_moment[entry["moment"]][entry["arm"]] = goodness(dim.key, value)
                    raw[entry["arm"]].append(value)
            if not raw:
                continue
            # The rubric scale is printed, because it is what a reader can check against the
            # anchors; the comparison underneath is in goodness units, where + is always better.
            summary = "  ".join(f"{arm} {statistics.fmean(raw[arm]):.2f}" for arm in arms if raw.get(arm))
            arrow = {"low": " (lower is better)", "mid": " (2 is best; compared as distance from 2)"}
            print(f"\n  {dim.key:<11} {summary}{arrow.get(BETTER.get(dim.key, 'high'), '')}")
            if BETTER.get(dim.key) == "mid":
                for arm in arms:
                    if not raw.get(arm):
                        continue
                    n = len(raw[arm])
                    share = [sum(1 for v in raw[arm] if round(v) == s) / n for s in (1, 2, 3)]
                    print(
                        f"    {arm:>8}  too short {share[0]:>4.0%}   right {share[1]:>4.0%}   "
                        f"too long {share[2]:>4.0%}"
                    )
            for arm in arms:
                if arm == args.baseline or not raw.get(arm):
                    continue
                n, mean, half, wins = paired(per_moment, arm, args.baseline)
                if n < 3:
                    continue
                verdict = "  --  " if abs(mean) <= half else ("BETTER" if mean > 0 else " WORSE")
                print(
                    f"    {arm:>8} vs {args.baseline:<8} {mean:+.2f} +/- {half:.2f}  "
                    f"wins {wins:.0%} of {n} moments  {verdict}"
                )

    # The probe's own view of the same turns, from the key rather than a re-run of the
    # encoder. Printed last and separately because it is the thing being checked, not a
    # measurement: these are the numbers GRPO was maximising.
    probe: dict[str, dict[str, list[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for entry in key.values():
        for name, value in (entry.get("probe") or {}).items():
            probe[name][entry["arm"]].append(value)
    if probe:
        print(f"\n{'=' * 78}\nPROBE, on all {len(key)} turns — what the reward saw, for comparison")
        for name, by_arm in probe.items():
            cells = "  ".join(f"{arm} {statistics.fmean(by_arm[arm]):.2f}" for arm in arms if by_arm.get(arm))
            print(f"  {name:<11} {cells}")
        if "total" in probe:
            totals["probe"] = {a: statistics.fmean(probe["total"][a]) for a in arms if probe["total"].get(a)}

    # Side by side, in gain-over-baseline rather than levels, because the panels do not share
    # a zero: the raters' total is a mean of 1-3 rubric scores and the probe's is a mean of
    # ridge outputs that were never calibrated to that scale. The levels are not comparable
    # and the movements are, which is the only claim being made.
    if len(totals) > 1 and args.baseline in next(iter(totals.values()), {}):
        print(f"\n{'=' * 78}\nGAIN OVER {args.baseline.upper()}, same scalarisation, three ways of measuring it")
        print(f"  {'panel':<10}" + "".join(f"{a:>12}" for a in arms if a != args.baseline))
        for panel, by_arm in totals.items():
            if args.baseline not in by_arm:
                continue
            row = "".join(f"{by_arm[a] - by_arm[args.baseline]:>+12.2f}" for a in arms if a != args.baseline and a in by_arm)
            print(f"  {panel:<10}{row}")
        print("\n  A probe gain much larger than the rated gain is the reward-hacking signature.")

    print("\nRead the two panels against each other. A dimension the probe moved and the")
    print("raters did not is the reward-hacking case, and it is the reason for the blinding.")
    if len(arms) > 2:
        print(f"Arm pairs other than vs-{args.baseline}: " + ", ".join(f"{a}/{b}" for a, b in itertools.combinations(arms, 2)))


if __name__ == "__main__":
    main()
