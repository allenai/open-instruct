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


def fit_mlp(xtr, ytr, xte, args):
    """An MLP given the same courtesies ridge gets, so the comparison means something.

    Ridge picks its penalty per fold out of twelve candidates. An MLP on one fixed
    alpha is not losing to ridge, it is losing to having been tuned, and reporting
    that as "the property is linear" would be reading a configuration error as a
    result. So: reduce to a rank the labels can support before adding capacity, and
    choose alpha on the network's own held-out split, which never sees the test fold.
    """
    import numpy as np  # noqa: PLC0415
    from sklearn.decomposition import PCA  # noqa: PLC0415
    from sklearn.neural_network import MLPRegressor  # noqa: PLC0415

    pca = PCA(n_components=min(args.components, len(xtr) - 1), random_state=args.seed).fit(xtr)
    ztr, zte = pca.transform(xtr), pca.transform(xte)
    best, best_score = None, -np.inf
    for alpha in (0.1, 1.0, 10.0, 100.0):
        mlp = MLPRegressor(
            hidden_layer_sizes=(args.hidden_units,),
            alpha=alpha,
            max_iter=2000,
            early_stopping=True,
            n_iter_no_change=30,
            validation_fraction=0.15,
            random_state=args.seed,
        ).fit(ztr, ytr)
        if mlp.best_validation_score_ > best_score:
            best, best_score = mlp, mlp.best_validation_score_
    return best.predict(zte)


def run_dimension(X, y, groups, args, models: tuple[str, ...] | None = None) -> dict:
    import numpy as np  # noqa: PLC0415
    from sklearn.linear_model import RidgeCV  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    names = list(models) if models else ["ridge"] + ([] if args.skip_mlp else ["mlp"])
    predictions = {name: np.zeros(len(y)) for name in names}
    for test_idx in folds(groups, args.folds, args.seed):
        train_idx = [i for i in range(len(y)) if i not in set(test_idx)]
        scaler = StandardScaler().fit(X[train_idx])
        xtr, xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        ridge = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(xtr, y[train_idx])
        predictions["ridge"][test_idx] = ridge.predict(xte)
        if "mlp" in predictions:
            predictions["mlp"][test_idx] = fit_mlp(xtr, y[train_idx], xte, args)
    scores = {name: pearson(list(map(float, p)), list(map(float, y))) for name, p in predictions.items()}
    return scores, predictions


def main() -> None:
    import numpy as np  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", required=True, help="npz from extract_hidden")
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--pooling", default="", choices=("", "mean", "last", "eot"), help="blank sweeps all three")
    parser.add_argument("--layer", type=int, default=-1, help="index INTO the stored layers; -1 sweeps all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-units", type=int, default=128, help="MLP width")
    parser.add_argument("--components", type=int, default=128, help="PCA rank in front of the MLP")
    parser.add_argument("--skip-mlp", action="store_true", help="ridge only; the sweep is 20x faster")
    parser.add_argument(
        "--reference", default="", help="a rater's label file; also score the probe against them alone"
    )
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
    questions, texts = read_slices(args, ids)
    reference: dict[str, dict] = {}
    without_reference = by_unit
    if args.reference:
        name = args.reference.split("/")[-1].removesuffix(".json")
        reference = {uid: rec[name] for uid, rec in by_unit.items() if name in rec}
        if not reference:
            raise SystemExit(f"no rater named {name!r} among {sorted({k for r in by_unit.values() for k in r})}")
        without_reference = {uid: {k: v for k, v in rec.items() if k != name} for uid, rec in by_unit.items()}

    layer_choices = range(len(stored_layers)) if args.layer < 0 else [args.layer]
    print(f"poolings={poolings}  layers={stored_layers}  units={len(ids)}")
    print("cross-validated r, folds grouped by question\n")
    header = f"  {'dimension':<12} {'n':>5} {'pooling':>8} {'layer':>6} {'surface':>8} {'ridge':>7}"
    if not args.skip_mlp:
        header += f" {'mlp':>7}"
    if args.ceilings:
        header += f" {'ceiling':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for dim in DIMENSIONS:
        scores = consensus(by_unit, dim.key)
        # Agents only, so the yardstick is not partly the labels it is scored against.
        agents = consensus(without_reference, dim.key) if reference else {}
        usable = [uid for uid in scores if uid in index]
        if len(usable) < 40:
            print(f"  {dim.key:<12} too few labelled units ({len(usable)})")
            continue
        y = np.array([scores[uid] for uid in usable], dtype=np.float32)
        groups = [questions.get(uid, uid) for uid in usable]
        rows = np.array([index[uid] for uid in usable])
        ceiling = agreement_ceiling(by_unit, dim.key) if args.ceilings else None

        surface = np.array([surface_features(texts.get(uid, "")) for uid in usable], dtype=np.float32)
        naive = run_dimension(surface, y, groups, args, models=("ridge",))[0]["ridge"]

        best = None
        for pooling in poolings:
            feats = blob[pooling].astype(np.float32)
            for li in layer_choices:
                result, preds = run_dimension(feats[rows, li, :], y, groups, args)
                score = max(result.values())
                if best is None or score > best[0]:
                    best = (score, pooling, stored_layers[li], result, preds)
                if args.verbose:
                    print(row(dim.key, len(y), pooling, stored_layers[li], result, naive))
        if best:
            _, pooling, layer, result, preds = best
            line = row(dim.key, len(y), pooling, layer, result, naive)
            if ceiling is not None:
                line += f" {ceiling:>8.2f}"
            print(line + ("   <- best cell" if args.verbose else ""))
            if reference:
                versus_human(dim.key, usable, preds["ridge"], y, reference, agents)

    print("\n  READ THE SURFACE COLUMN FIRST. Eight features - length, question marks,")
    print("  digits - with no idea what teaching is. Where it matches ridge, the label")
    print("  is a property of the text's shape and the hidden states bought nothing.")
    print("  Only the gap between surface and ridge is evidence about pedagogy.")
    print("\n  Best cell per dimension, chosen over poolings x layers. That selection is")
    print("  itself fitted, so the number is optimistic; treat a close second as a tie.")
    print("  Ceilings use Spearman-Brown for a k-rater mean, not the single-rater bound.")


def versus_human(key: str, usable: list[str], pred, y, reference: dict, agents: dict) -> None:
    """Does the probe track the human, or only the agents it was distilled from?

    The target is a consensus of six models and one person, so the models decide it.
    A probe can fit that consensus perfectly while tracking something the person does
    not recognise, and every number above would look the same. The comparison that
    settles it is against the human alone - with the agents' own score on the same
    units as the yardstick, because the probe cannot be expected to beat its teachers.
    """
    rows = [
        (i, reference[uid][key]) for i, uid in enumerate(usable) if isinstance(reference.get(uid, {}).get(key), int)
    ]
    if len(rows) < 12:
        print(f"     vs human: only {len(rows)} labelled units, not reporting")
        return
    idx = [i for i, _ in rows]
    human = [float(v) for _, v in rows]
    probe_r = pearson([float(pred[i]) for i in idx], human)
    agent_r = pearson([float(agents[usable[i]]) for i in idx], human)
    print(f"     vs human (n={len(rows)}): probe {probe_r:.2f}, the agents themselves {agent_r:.2f}")


def surface_features(text: str) -> list[float]:
    """Things nobody would call teaching: length, punctuation, digits.

    The control that decides whether this project is measuring anything. A hidden
    state predicts the length of its own turn almost perfectly, so a dimension that
    is secretly length will score near 1.0 and look like a triumph. If these eight
    numbers match the 4096, the labels are about form and the states are not earning
    their cost; only the gap between the two columns is evidence about pedagogy.
    """
    words = text.split()
    n_words = len(words) or 1
    sentences = sum(text.count(c) for c in ".!?") or 1
    return [
        math.log1p(len(words)),
        math.log1p(len(text)),
        text.count("?"),
        float("?" in text),
        sentences,
        sum(c.isdigit() for c in text) / len(text or " "),
        sum(len(w) for w in words) / n_words,
        len(words) / sentences,
    ]


def row(key: str, n: int, pooling: str, layer, result: dict, naive: float) -> str:
    line = f"  {key:<12} {n:>5} {pooling:>8} {layer:>6} {naive:>8.2f} {result['ridge']:>7.2f}"
    if "mlp" in result:
        line += f" {result['mlp']:>7.2f}"
    return line


def read_slices(args, ids: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """unit id -> question, and unit id -> tutor turn."""
    questions: dict[str, str] = {}
    texts: dict[str, str] = {}
    for path in itertools.chain.from_iterable(glob.glob(p) or [p] for p in args.slices):
        try:
            with open(path) as handle:
                for unit in json.load(handle).get("units", []):
                    questions[unit["id"]] = unit["question"]
                    texts[unit["id"]] = unit.get("tutor_turn", "")
        except (OSError, json.JSONDecodeError):
            continue
    missing = [i for i in ids if i not in questions]
    if missing:
        print(f"  note: {len(missing)} units have no question on file; they fold as singletons")
    return questions, texts


def agreement_ceiling(by_unit: dict[str, dict[str, dict]], key: str) -> float:
    """Most any predictor could correlate with these labels, from rater reliability.

    The target is a MEAN of k raters, not one rater, and averaging cancels noise, so
    the single-rater bound is the wrong one - it is why the first run reported every
    probe as beating its ceiling, which is not a thing that can happen. Spearman-Brown
    converts pairwise agreement into the reliability of the k-rater mean, and the
    ceiling on correlating with a target is the square root of its reliability.
    """
    pairs, counts = [], []
    for records in by_unit.values():
        scores = [r[key] for r in records.values() if isinstance(r.get(key), int)]
        if scores:
            counts.append(len(scores))
        pairs.extend((x, y) for x, y in itertools.combinations(scores, 2))
    if len(pairs) < 10 or not counts:
        return float("nan")
    r = pearson([float(a) for a, _ in pairs], [float(b) for _, b in pairs])
    if r != r or r <= 0:
        return float("nan")
    k = statistics.median(counts)
    reliability = k * r / (1 + (k - 1) * r)
    return math.sqrt(min(reliability, 1.0))


if __name__ == "__main__":
    main()
