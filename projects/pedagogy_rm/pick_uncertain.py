"""Choose the few turns worth a human's time, out of the many that are not.

    python projects/pedagogy_rm/pick_uncertain.py \
        --pool data/eval50/pool.json \
        --labels 'data/eval50/labels/agent_*.json' \
        --already data/eval50/labels/sophia.json \
        --key data/eval50/key.json --n 40 --out data/eval50/round2/pool.json

WHY NOT JUST LABEL MORE AT RANDOM. Most turns are easy: six raters give them the same score,
the probe agrees, and a human label confirms what everyone already knew. Those cost the same
minute as a hard one and move the fit by nothing. The information is concentrated in the
turns where the raters split, and a sample drawn there is worth several times its size -
which is the difference between labelling forty and labelling all of them.

TWO KINDS OF DISAGREEMENT, AND THEY ARE NOT THE SAME KIND OF USEFUL.

`spread` is how much the six raters differ from each other on a turn. High spread means the
rubric genuinely underdetermines this case, so a human ruling is what resolves it. This is
classic uncertainty sampling.

`gap` is how far the agent consensus sits from what the probe already predicts. High gap
means the head and the raters have different opinions, and one of them is wrong; a label
there either corrects the head or exonerates it. This is the term that makes the sample
useful for *refitting* rather than only for measuring.

They are summed after each is scaled to its own spread across the pool, because their raw
units are unrelated - a standard deviation of ratings and a difference of predicted scores -
and whichever happened to be numerically larger would otherwise decide the ranking alone.

STRATIFIED BY ARM, BECAUSE THE HARD TURNS ARE NOT SPREAD EVENLY. The trained arms produce
shorter, more uniform turns, so left alone this ranking would fill up with base-policy turns
and the refit would learn least about the policies it is meant to score. Equal quotas per arm
cost a little ranking purity and buy a sample that covers the thing being measured.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics


def consensus_and_spread(raters: list[dict], keys: list[str]) -> tuple[dict[str, float], float]:
    """Per-dimension mean, and the mean across dimensions of the rater standard deviation."""
    means, sds = {}, []
    for key in keys:
        votes = [r[key] for r in raters if isinstance(r.get(key), int)]
        if not votes:
            continue
        means[key] = statistics.fmean(votes)
        sds.append(statistics.pstdev(votes) if len(votes) > 1 else 0.0)
    return means, (statistics.fmean(sds) if sds else 0.0)


def scaled(values: dict[str, float]) -> dict[str, float]:
    """Divided by its own standard deviation, so two unrelated units can be added."""
    if len(values) < 2:
        return dict.fromkeys(values, 0.0)
    sd = statistics.pstdev(values.values()) or 1.0
    mean = statistics.fmean(values.values())
    return {k: (v - mean) / sd for k, v in values.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", default="data/eval50/pool.json")
    parser.add_argument("--labels", nargs="+", default=["data/eval50/labels/agent_*.json"])
    parser.add_argument("--already", default="", help="labels you have; those turns are skipped")
    parser.add_argument("--key", default="data/eval50/key.json", help="for the probe scores and the arms")
    parser.add_argument("--dimensions", default="leak,targeted,actionable,elicits,length_fit,correct")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--out", default="data/eval50/round2/pool.json")
    args = parser.parse_args()

    keys = args.dimensions.split(",")
    with open(args.pool) as handle:
        units = {u["id"]: u for u in json.load(handle)["units"]}
    with open(args.key) as handle:
        key = json.load(handle)["key"]

    done: set[str] = set()
    if args.already:
        with open(args.already) as handle:
            done = {r["id"] for r in json.load(handle)["labels"] if all(k in r for k in keys)}

    by_unit: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted({p for pattern in args.labels for p in (glob.glob(pattern) or [pattern])}):
        with open(path) as handle:
            for record in json.load(handle).get("labels", []):
                by_unit[record["id"]].append(record)

    candidates = [u for u in units if u not in done and by_unit.get(u)]
    spread, gap, means = {}, {}, {}
    for uid in candidates:
        mean, sd = consensus_and_spread(by_unit[uid], keys)
        means[uid] = mean
        spread[uid] = sd
        probe = (key.get(uid) or {}).get("probe") or {}
        shared = [k for k in mean if k in probe]
        gap[uid] = statistics.fmean([abs(mean[k] - probe[k]) for k in shared]) if shared else 0.0

    z_spread, z_gap = scaled(spread), scaled(gap)
    score = {u: z_spread[u] + z_gap[u] for u in candidates}

    # Equal quotas per arm, filled in score order, so the sample covers what it measures.
    arms = collections.defaultdict(list)
    for uid in sorted(candidates, key=lambda u: -score[u]):
        arms[(key.get(uid) or {}).get("arm", "?")].append(uid)
    quota = max(1, args.n // max(1, len(arms)))
    chosen: list[str] = []
    for arm in sorted(arms):
        chosen.extend(arms[arm][:quota])
    for uid in sorted(candidates, key=lambda u: -score[u]):  # top up if a quota fell short
        if len(chosen) >= args.n:
            break
        if uid not in chosen:
            chosen.append(uid)

    # Written most-informative-first, and therefore not grouped by arm, for two reasons.
    # A labelling session is abandoned rather than finished, so the order decides what is
    # actually collected: quitting at fifteen should leave the fifteen best, not the whole of
    # one arm. And filling per-arm quotas in turn would emit fourteen consecutive turns from
    # the same policy, which is a pattern a labeller can notice in a pool that is supposed to
    # be blind.
    chosen.sort(key=lambda u: -score[u])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump({"units": [units[u] for u in chosen]}, handle, indent=1)

    print(f"{len(units)} turns, {len(done)} already labelled, {len(candidates)} candidates")
    print(f"picked {len(chosen)}: " + "  ".join(f"{a} {sum(1 for u in chosen if key.get(u, {}).get('arm') == a)}" for a in sorted(arms)))
    print(f"\n{'':>4} {'rater spread':>13} {'probe gap':>10}")
    for label, group in (("picked", chosen), ("rest", [u for u in candidates if u not in chosen])):
        if group:
            print(f"{label:>6} {statistics.fmean(spread[u] for u in group):>11.2f} {statistics.fmean(gap[u] for u in group):>10.2f}")
    print(f"\nwrote {args.out}")
    print("Label it with label_ui.py --units " + args.out)


if __name__ == "__main__":
    main()
