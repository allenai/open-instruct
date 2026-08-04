"""Two sampled policies, side by side: did the probe rise for a reason?

    python projects/pedagogy_rm/compare_policies.py \
        --before data/samples/base.json --after data/samples/trained.json

THE ONE QUESTION THIS PROJECT HAS TO ANSWER. A GRPO run against a dense continuous
reward will almost certainly make that reward go up, so "reward went up" is close to
uninformative on its own. There are three ways it can go up and only one of them is
the result:

  1. the policy teaches better                     -> what we want
  2. the policy memorised these questions          -> the prompts here are held out
                                                      by question, so this shows as a
                                                      flat curve on unseen ones
  3. the policy found what this head likes         -> invisible in every number the
                                                      head computes, which is why the
                                                      surface table and the agents
                                                      below exist

WHAT THE SURFACE TABLE IS FOR. probe.py measured that eight features with no notion of
teaching predict `concise` at 0.96 and `elicits` at 0.81. So a reward rise that arrives
together with a large move in mean length, question-mark rate or digit density is
consistent with the policy having found the surface feature rather than the pedagogy.
The deltas are printed in units of the before-policy's own standard deviation, because
a shift of 0.4 words means nothing and a shift of two standard deviations means a great
deal, and only the second is visible in that form.

WHAT THE AGENT COLUMN IS FOR. The probe cannot referee itself. If label_agents.py has
rated either file - the units it writes are the units this reads - the ratings are shown
beside the probe's. Their agreement with the human labeller ran 0.72 to 0.92 by
dimension, so a probe rise the agents do not see is the strongest evidence available
here that the number was gamed.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics

from projects.pedagogy_rm.probe import surface_features

SURFACE_NAMES = (
    "log words",
    "log chars",
    "question marks",
    "asks anything",
    "sentences",
    "digit density",
    "word length",
    "words a sentence",
)


def load(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def agent_ratings(units: list[dict], pattern: str) -> dict[str, dict[str, float]]:
    """Mean rating per unit per dimension, over whichever raters covered it.

    Keyed by unit id rather than by position, because a rater that failed on one unit
    leaves a file shorter than the others and a positional join would silently attribute
    one turn's rating to another.
    """
    by_unit: dict[str, dict[str, list[float]]] = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            for uid, scores in json.load(handle).items():
                for dim, value in scores.items():
                    if isinstance(value, int | float):
                        by_unit.setdefault(uid, {}).setdefault(dim, []).append(float(value))
    ids = {unit["id"] for unit in units}
    return {
        uid: {dim: statistics.fmean(values) for dim, values in dims.items()}
        for uid, dims in by_unit.items()
        if uid in ids
    }


def summarise(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        return (values[0] if values else math.nan, math.nan)
    return statistics.fmean(values), statistics.stdev(values)


def row(name: str, before: list[float], after: list[float]) -> str:
    """One line of a comparison, with the change expressed against the spread it moved in.

    Half a standard deviation is where the arrow appears. Not a significance test - with a
    couple of hundred turns almost any real shift clears one of those - but the size at
    which a reader should want an explanation for the move.
    """
    b_mean, b_sd = summarise(before)
    a_mean, _ = summarise(after)
    delta = a_mean - b_mean
    effect = delta / b_sd if b_sd and not math.isnan(b_sd) else math.nan
    flag = "  <-- moved" if not math.isnan(effect) and abs(effect) >= 0.5 else ""
    return f"  {name:17} {b_mean:+8.3f} {a_mean:+8.3f} {delta:+8.3f} {effect:+7.2f}{flag}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--before", required=True, help="a sample_policy.py file for the untrained policy")
    parser.add_argument("--after", required=True, help="the same for the trained one")
    parser.add_argument("--labels", default="data/labels/samples_*.json", help="agent ratings, if any exist")
    args = parser.parse_args()

    before, after = load(args.before), load(args.after)
    b_units, a_units = before["units"], after["units"]
    dims = before.get("dimensions") or []
    print(f"before: {len(b_units)} turns from {before.get('tag') or args.before}")
    print(f"after:  {len(a_units)} turns from {after.get('tag') or args.after}")

    if dims and all("probe" in unit for unit in b_units + a_units):
        print("\nPROBE — what the policy was trained to raise")
        print(f"  {'dimension':17} {'before':>8} {'after':>8} {'delta':>8} {'sd':>7}")
        for dim in [*dims, "total"]:
            print(row(dim, [u["probe"][dim] for u in b_units], [u["probe"][dim] for u in a_units]))
    else:
        print("\nno probe scores; run `sample_policy.py score` on both files first")

    print("\nSURFACE — what nobody would call teaching")
    print(f"  {'feature':17} {'before':>8} {'after':>8} {'delta':>8} {'sd':>7}")
    b_surface = [surface_features(u["tutor_turn"]) for u in b_units]
    a_surface = [surface_features(u["tutor_turn"]) for u in a_units]
    for index, name in enumerate(SURFACE_NAMES):
        print(row(name, [f[index] for f in b_surface], [f[index] for f in a_surface]))

    b_agents = agent_ratings(b_units, args.labels)
    a_agents = agent_ratings(a_units, args.labels)
    if b_agents and a_agents:
        print("\nAGENTS — the referee the probe cannot be")
        print(f"  {'dimension':17} {'before':>8} {'after':>8} {'delta':>8} {'sd':>7}")
        rated = sorted({dim for dims_ in list(b_agents.values()) + list(a_agents.values()) for dim in dims_})
        for dim in rated:
            b_values = [scores[dim] for scores in b_agents.values() if dim in scores]
            a_values = [scores[dim] for scores in a_agents.values() if dim in scores]
            if b_values and a_values:
                print(row(dim, b_values, a_values))
        print(f"  on {len(b_agents)} before and {len(a_agents)} after")
    else:
        found = os.path.dirname(args.labels) or "."
        print(f"\nno agent ratings under {args.labels}; the probe is unrefereed until there are")
        print("  projects/pedagogy_rm/label_agents.py --units <a sample file> --out-dir " + found)

    print(
        "\nRead the three blocks together. A probe rise with a flat surface table and an "
        "agent rise is the result.\nA probe rise with a moved surface row is the surface "
        "feature. A probe rise the agents do not see is\nthe head being gamed, and neither "
        "of the last two is a teaching improvement."
    )


if __name__ == "__main__":
    main()
