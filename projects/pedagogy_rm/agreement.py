"""Per-dimension rater agreement. THE DECISION GATE.

    python -m projects.pedagogy_rm.agreement --labels data/labels/*.json

Run this the moment labels exist, before extracting a single hidden state.

WHY IT COMES FIRST. A probe cannot beat the noise in its own labels. If two
careful raters disagree about a dimension, the dimension is not well posed, and
no volume of labelling, no bigger model and no better architecture fixes it -
the ceiling is set by the question, not by the method. The previous project
learned this the expensive way: a ridge probe reached 0.581 predicting holistic
"goodness", which looked like a representation failure until agreement on that
same scale turned out to be 39% exact.

WHAT TO DO WITH THE NUMBER. Weighted kappa is the statistic to read, because it
gives partial credit for being one point off and corrects for the agreement two
raters would reach by chance alone - raw percentages flatter a scale where
everyone says 2.

    kappa >= 0.6     keep. Raters mean the same thing.
    0.4 - 0.6        keep, but the ceiling is low; report probe results against
                     it rather than against 1.0.
    < 0.4            DROP the dimension, or rewrite it and re-label a slice.
                     Training a probe on it wastes the labels you already paid
                     for and produces a reward that is mostly rater noise.

A high flag rate is the same signal arriving more politely: raters are telling
you the question does not fit the turns.

CEILING ON THE PROBE. The last column converts agreement into the correlation a
perfect probe could reach against these labels, which is what probe.py should be
compared with. Beating it is not possible; approaching it means the probe is
done and the labels are the limit.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import math
import statistics

from projects.pedagogy_rm.rubric import DIMENSIONS


def weighted_kappa(a: list[int], b: list[int], lo: int, hi: int) -> float:
    """Linearly weighted Cohen's kappa over an ordinal scale.

    Linear rather than quadratic weights: quadratic is conventional but forgives
    one-point disagreements so heavily that a scale nobody agrees on can still
    score well, which is the opposite of what this is for.
    """
    cats = list(range(lo, hi + 1))
    index = {c: i for i, c in enumerate(cats)}
    n = len(a)
    if n == 0:
        return float("nan")
    k = len(cats)
    span = max(k - 1, 1)

    observed = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        observed[index[x]][index[y]] += 1 / n
    count_a = collections.Counter(a)
    count_b = collections.Counter(b)

    num = den = 0.0
    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            weight = abs(i - j) / span
            expected = (count_a[ci] / n) * (count_b[cj] / n)
            num += weight * observed[i][j]
            den += weight * expected
    return 1.0 - num / den if den else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")


def load(paths: list[str], shots: dict[str, set[str]] | None = None) -> dict[str, dict[str, dict]]:
    """``{unit_id: {rater: record}}`` from one file per rater.

    Fills ``shots`` with ``{dimension: {unit_id}}`` for units an agent was shown
    the answer to, per dimension, so a partial re-rate does not needlessly
    disqualify units from validating the dimensions it left alone.
    """
    by_unit: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for path in paths:
        with open(path) as handle:
            blob = json.load(handle)
        if shots is not None and isinstance(blob, dict):
            stored = blob.get("shots") or {}
            if isinstance(stored, list):  # older files predate the per-dimension form
                stored = {d.key: stored for d in DIMENSIONS}
            for key, ids in stored.items():
                shots.setdefault(key, set()).update(ids)
        rater = blob.get("rater") or path.rsplit("/", 1)[-1].removesuffix(".json")
        records = blob.get("labels") if isinstance(blob, dict) else blob
        if isinstance(records, dict):
            records = [{"id": k, **v} for k, v in records.items()]
        for record in records or []:
            by_unit[record["id"]][rater] = record
    return dict(by_unit)


def against_reference(by_unit: dict[str, dict[str, dict]], reference: str, contaminated: set[str]) -> None:
    """Each rater versus one designated rater, plus their consensus versus it.

    Inter-agent agreement and agreement-with-the-human answer different
    questions, and only the second one licenses using agent labels. Six models
    can agree closely because they share training data and a house style; that
    is consistency, not accuracy. The consensus row is the one to read, because
    the mean of the agents is what a probe would actually be trained on.
    """
    raters = sorted({r for records in by_unit.values() for r in records} - {reference})
    shared = {u: recs for u, recs in by_unit.items() if reference in recs}
    # A unit counts for a dimension only when the reference actually scored it.
    # Single-dimension labelling sessions leave the others blank, and counting
    # those would advertise power the comparison does not have.
    clean = {
        d.key: {
            u
            for u in shared
            if u not in contaminated.get(d.key, set()) and isinstance(shared[u][reference].get(d.key), int)
        }
        for d in DIMENSIONS
    }
    counts = {d.key: len(clean[d.key]) for d in DIMENSIONS}
    if not any(counts.values()):
        print("\n  NO CLEAN UNITS. Every unit you labelled was used to calibrate the agents,")
        print("  so there is nothing left to check them against. Label some more in the UI:")
        print("  the agents have already run, so anything you label now is untainted by")
        print("  construction. Fifteen or so is enough to see whether they track you.")
        return
    print(f"\n  Against '{reference}', excluding each dimension's own few-shot examples — kappa_w")
    print("  n differs per dimension because they were not all calibrated on the same units.")
    header = "  " + f"{'rater':<11}" + "".join(f"{d.key:>11}" for d in DIMENSIONS)
    print(header)
    print("  " + f"{'(n)':<11}" + "".join(f"{counts[d.key]:>11}" for d in DIMENSIONS))
    print("  " + "-" * (len(header) - 2))

    for rater in [*raters, "CONSENSUS"]:
        cells = []
        for dim in DIMENSIONS:
            a, b = [], []
            for unit_id in clean[dim.key]:
                recs = shared[unit_id]
                truth = recs[reference].get(dim.key)
                if not isinstance(truth, int):
                    continue
                if rater == "CONSENSUS":
                    votes = [recs[r][dim.key] for r in raters if r in recs and isinstance(recs[r].get(dim.key), int)]
                    if not votes:
                        continue
                    # Rounded mean, so the consensus stays on the rubric's scale
                    # and can be compared with a single rater on equal terms.
                    value = round(statistics.fmean(votes))
                elif rater in recs and isinstance(recs[rater].get(dim.key), int):
                    value = recs[rater][dim.key]
                else:
                    continue
                a.append(truth)
                b.append(value)
            cells.append(weighted_kappa(a, b, dim.lo, dim.hi) if len(a) >= 4 else float("nan"))
        marker = "  <- what a probe would learn" if rater == "CONSENSUS" else ""
        print("  " + f"{rater:<11}" + "".join(f"{c:>11.2f}" for c in cells) + marker)

    print("\n  A dimension where the agents agree with EACH OTHER but not with you is")
    print("  the dangerous case: it looks reliable and is measuring something else.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", nargs="+", required=True, help="one json per rater; globs allowed")
    parser.add_argument("--reference", default="", help="rater name to score everyone else against, e.g. sophia")
    args = parser.parse_args()

    paths = sorted(set(itertools.chain.from_iterable(glob.glob(p) or [p] for p in args.labels)))
    shots: dict[str, set[str]] = {}
    by_unit = load(paths, shots)
    shared = {u: r for u, r in by_unit.items() if len(r) >= 2}
    print(f"{len(paths)} raters, {len(by_unit)} units, {len(shared)} rated by 2+")
    if not shared:
        raise SystemExit(
            "no overlapping units - agreement is unmeasurable.\n"
            "Rebuild slices with --overlap > 0; this is the mistake the last round made."
        )

    flags = sum(1 for records in by_unit.values() for r in records.values() if r.get("flag"))
    total = sum(len(r) for r in by_unit.values())
    print(f"flagged {flags}/{total} ({flags / total:.0%}) — a high rate means the rubric does not fit the turns\n")

    header = f"  {'dimension':<12} {'n':>4} {'exact':>7} {'within1':>8} {'kappa_w':>8} {'r':>7}  {'probe ceiling':>13}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    verdicts = {}
    for dim in DIMENSIONS:
        pairs = []
        for records in shared.values():
            scores = [r[dim.key] for r in records.values() if isinstance(r.get(dim.key), int)]
            pairs.extend((x, y) for x, y in itertools.combinations(scores, 2))
        if not pairs:
            print(f"  {dim.key:<12} {'-':>4}  (no overlapping ratings)")
            continue
        a = [x for x, _ in pairs]
        b = [y for _, y in pairs]
        exact = sum(x == y for x, y in pairs) / len(pairs)
        within1 = sum(abs(x - y) <= 1 for x, y in pairs) / len(pairs)
        kappa = weighted_kappa(a, b, dim.lo, dim.hi)
        r = pearson([float(x) for x in a], [float(x) for x in b])
        # A probe predicts the consensus, whose reliability is the correlation
        # between two raters; the most any predictor can correlate with a noisy
        # label is sqrt of that reliability.
        ceiling = math.sqrt(max(r, 0.0)) if r == r else float("nan")
        print(
            f"  {dim.key:<12} {len(pairs):>4} {exact:>6.0%} {within1:>7.0%} {kappa:>8.2f} {r:>7.2f}  {ceiling:>13.2f}"
        )
        verdicts[dim.key] = kappa

    print()
    keep = [k for k, v in verdicts.items() if v >= 0.4]
    drop = [k for k, v in verdicts.items() if v < 0.4]
    if drop:
        print(f"  DROP or rewrite: {', '.join(drop)}")
        print("     Raters do not mean the same thing. A probe trained here learns rater noise.")
    if keep:
        print(f"  Keep: {', '.join(keep)}")
        print("     Compare probe.py against the ceiling column, not against 1.0.")
    if not keep:
        print("  No dimension survives. Rewrite the rubric before labelling more.")

    if args.reference:
        present = {r for records in by_unit.values() for r in records}
        if args.reference not in present:
            raise SystemExit(f"no rater named '{args.reference}'; have {sorted(present)}")
        against_reference(by_unit, args.reference, shots)


if __name__ == "__main__":
    main()
