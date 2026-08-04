"""Can an MLP read each pedagogy dimension out of OLMo's hidden states?

    python -m projects.pedagogy_rm.probe \
        --hidden data/hidden.npz --labels 'data/labels/*.json' --ceilings

READ THE RESULT AGAINST THE CEILING, NOT AGAINST ZERO. A probe cannot correlate
with a noisy label better than the label's own reliability allows; agreement.py
prints that bound per dimension. A probe at r=0.55 against a ceiling of 0.60 has
essentially solved the problem and the labels are now the limit. The same 0.55
against a ceiling of 0.95 means the representation is not carrying the property.
Reported against zero, those two look identical, and the previous project's
0.581 was read exactly that way.

A LINEAR BASELINE RUNS ALONGSIDE, ALWAYS. An MLP that does not beat ridge is an
MLP that is not earning its extra capacity or its extra ways to fool you, and on
a few hundred labels that happens often. If ridge wins, use ridge - it is faster
inside an RL loop, which is the entire point of this project.

GROUPED SPLITS. Folds are split by QUESTION, not by turn. Turns from the same
item share a question, and a probe that has seen one of them can recognise the
other from surface features; a random split would report that as skill.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import statistics

from projects.pedagogy_rm.agreement import load as load_labels
from projects.pedagogy_rm.agreement import pearson
from projects.pedagogy_rm.rubric import DIMENSIONS


def consensus(by_unit: dict[str, dict[str, dict]], key: str) -> dict[str, float]:
    """Mean score per unit. Averaging is the cheapest noise reduction available."""
    out = {}
    for uid, records in by_unit.items():
        scores = [r[key] for r in records.values() if isinstance(r.get(key), int)]
        if scores:
            out[uid] = statistics.fmean(scores)
    return out


def folds(groups: list[str], k: int, seed: int = 0) -> list[list[int]]:
    """k folds that never split a question across train and test."""
    import random  # noqa: PLC0415

    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    assignment = {g: i % k for i, g in enumerate(unique)}
    out: list[list[int]] = [[] for _ in range(k)]
    for i, g in enumerate(groups):
        out[assignment[g]].append(i)
    return out


def run_dimension(X, y, groups, args) -> dict:
    import numpy as np  # noqa: PLC0415
    from sklearn.linear_model import RidgeCV  # noqa: PLC0415
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    predictions = {"ridge": np.zeros(len(y)), "mlp": np.zeros(len(y))}
    for test_idx in folds(groups, args.folds, args.seed):
        train_idx = [i for i in range(len(y)) if i not in set(test_idx)]
        scaler = StandardScaler().fit(X[train_idx])
        xtr, xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        ridge = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(xtr, y[train_idx])
        predictions["ridge"][test_idx] = ridge.predict(xte)
        mlp = MLPRegressor(
            hidden_layer_sizes=(args.hidden_units,),
            alpha=args.alpha,
            max_iter=1500,
            early_stopping=True,
            n_iter_no_change=25,
            random_state=args.seed,
        ).fit(xtr, y[train_idx])
        predictions["mlp"][test_idx] = mlp.predict(xte)
    return {name: pearson(list(map(float, p)), list(map(float, y))) for name, p in predictions.items()}


def main() -> None:
    import numpy as np  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", required=True, help="npz from extract_hidden")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--pooling", default="", choices=("", "mean", "last", "eot"), help="blank sweeps all three")
    parser.add_argument("--layer", type=int, default=-1, help="index INTO the stored layers; -1 sweeps all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-units", type=int, default=256, help="MLP width")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ceilings", action="store_true", help="also print the agreement bound per dimension")
    parser.add_argument("--verbose", action="store_true", help="every pooling x layer cell, not just the best")
    parser.add_argument(
        "--slices",
        nargs="+",
        default=["data/label_slices/slice_*.json"],
        help="slice files, read only to group folds by question",
    )
    args = parser.parse_args()

    blob = np.load(args.hidden, allow_pickle=False)
    ids = [str(x) for x in blob["ids"]]
    stored_layers = list(blob["layers"])
    poolings = [args.pooling] if args.pooling else [p for p in ("eot", "last", "mean") if p in blob]
    index = {uid: i for i, uid in enumerate(ids)}

    paths = sorted(set(itertools.chain.from_iterable(glob.glob(p) or [p] for p in args.labels)))
    by_unit = load_labels(paths)
    questions = question_map(args, ids)

    layer_choices = range(len(stored_layers)) if args.layer < 0 else [args.layer]
    print(f"poolings={poolings}  layers={stored_layers}  units={len(ids)}")
    print("cross-validated r, folds grouped by question\n")
    header = f"  {'dimension':<12} {'n':>5} {'pooling':>8} {'layer':>6} {'ridge':>7} {'mlp':>7}"
    if args.ceilings:
        header += f" {'ceiling':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dim in DIMENSIONS:
        scores = consensus(by_unit, dim.key)
        usable = [uid for uid in scores if uid in index]
        if len(usable) < 40:
            print(f"  {dim.key:<12} too few labelled units ({len(usable)})")
            continue
        y = np.array([scores[uid] for uid in usable], dtype=np.float32)
        groups = [questions.get(uid, uid) for uid in usable]
        rows = np.array([index[uid] for uid in usable])
        ceiling = agreement_ceiling(by_unit, dim.key) if args.ceilings else None

        best = None
        for pooling in poolings:
            feats = blob[pooling].astype(np.float32)
            for li in layer_choices:
                result = run_dimension(feats[rows, li, :], y, groups, args)
                score = max(result.values())
                if best is None or score > best[0]:
                    best = (score, pooling, stored_layers[li], result)
                if args.verbose:
                    print(
                        f"  {dim.key:<12} {len(y):>5} {pooling:>8} {stored_layers[li]:>6} "
                        f"{result['ridge']:>7.2f} {result['mlp']:>7.2f}"
                    )
        if best:
            _, pooling, layer, result = best
            line = f"  {dim.key:<12} {len(y):>5} {pooling:>8} {layer:>6} {result['ridge']:>7.2f} {result['mlp']:>7.2f}"
            if ceiling is not None:
                line += f" {ceiling:>8.2f}"
            print(line + ("   <- best cell" if args.verbose else ""))

    print("\n  Best cell per dimension, chosen over poolings x layers. That selection is")
    print("  itself fitted, so the number is optimistic; treat a close second as a tie.")
    print("  A probe near its ceiling means the labels are the limit, not the states.")
    print("  If ridge matches the MLP, ship ridge: it is cheaper inside the RL loop.")


def question_map(args, ids: list[str]) -> dict[str, str]:
    """unit id -> question, so folds can be grouped by item rather than by turn."""
    mapping: dict[str, str] = {}
    for path in itertools.chain.from_iterable(glob.glob(p) or [p] for p in args.slices):
        try:
            with open(path) as handle:
                for unit in json.load(handle).get("units", []):
                    mapping[unit["id"]] = unit["question"]
        except (OSError, json.JSONDecodeError):
            continue
    missing = [i for i in ids if i not in mapping]
    if missing:
        print(f"  note: {len(missing)} units have no question on file; they fold as singletons")
    return mapping


def agreement_ceiling(by_unit: dict[str, dict[str, dict]], key: str) -> float:
    """Most any predictor could correlate with these labels, from rater reliability."""
    pairs = []
    for records in by_unit.values():
        scores = [r[key] for r in records.values() if isinstance(r.get(key), int)]
        pairs.extend((x, y) for x, y in itertools.combinations(scores, 2))
    if len(pairs) < 10:
        return float("nan")
    r = pearson([float(a) for a, _ in pairs], [float(b) for _, b in pairs])
    return math.sqrt(max(r, 0.0)) if r == r else float("nan")


if __name__ == "__main__":
    main()
