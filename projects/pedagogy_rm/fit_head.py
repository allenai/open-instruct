"""Fit the shipping head on everything, and write it out with its scaler.

    python -m projects.pedagogy_rm.fit_head --out data/head.npz

Ridge, because a tuned MLP lost to it on all five dimensions and a dot product
costs nothing inside a rollout. One (pooling, layer) per dimension, taken from
probe.py's sweep rather than chosen here, so this file cannot quietly disagree
with the numbers in the README.

THE ATTACKS GO IN, FOR LEAK ONLY. Without them the leak head misses 93% of turns
that hand over the answer at the start, which is the failure that matters once a
policy is optimising against it. They are not applied to the other dimensions
because appending a sentence changes those in ways nobody has labelled - it makes
the turn longer, so `concise` is now wrong, and there is no honest label to give
it. Augmenting a dimension with guessed labels would poison it to fix nothing.

CONCISE IS SAVED BUT NOT RECOMMENDED. Eight surface features predict it at 0.96
against the states' 0.97, so it is a word counter, and a policy rewarded on it
learns to be brief and nothing else. It is written out so the decision stays with
whoever builds the reward, and flagged so the decision is deliberate.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics

from projects.pedagogy_rm.rubric import DIMENSIONS

#: (pooling, layer) per dimension, from probe.py's sweep over three poolings and
#: seven layers. Layer 16 of 32 wins nearly everywhere: the last layer is
#: specialised for predicting the next token rather than for summarising.
CELLS = {
    "leak": ("last", 16),
    "targeted": ("mean", 16),
    "actionable": ("last", 20),
    "elicits": ("last", 16),
    "concise": ("eot", 12),
}
SURFACE_BOUND = {"concise": 0.96}  # dimensions a bag of trivial features already predicts


def choose_attacks(hack, n_real: int, ratio: float, seed: int) -> list[int]:
    """A minority of attacks, spread evenly over phrasing and position.

    All 1704 of them against 600 real turns is 74% of the training rows carrying
    the same label, and the head saturates: every prediction pinned at 3, zero
    variance, and a reward that is a constant. The validated run used 212 against
    ~300, so the attacks are capped at a fraction of the real data and sampled
    across the (phrasing, position) cells rather than taken in file order, which
    would have loaded up on whichever phrasing happens to come first.
    """
    import random  # noqa: PLC0415

    cells: dict[tuple[str, str], list[int]] = {}
    for i, uid in enumerate(str(x) for x in hack["ids"]):
        _, variant, where = uid.rsplit(":", 2)
        cells.setdefault((variant, where), []).append(i)
    budget = max(1, int(ratio * n_real))
    per_cell = max(1, budget // max(len(cells), 1))
    rng = random.Random(seed)
    chosen: list[int] = []
    for rows in cells.values():
        chosen.extend(rng.sample(rows, min(per_cell, len(rows))))
    return sorted(chosen)


def main() -> None:
    import numpy as np  # noqa: PLC0415
    from sklearn.linear_model import RidgeCV  # noqa: PLC0415
    from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

    from projects.pedagogy_rm.agreement import load as load_labels  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hidden", default="data/hidden.npz")
    parser.add_argument("--hack", default="data/hack_hidden.npz", help="adversarial states; leak only")
    parser.add_argument("--labels", default="data/labels/*.json")
    parser.add_argument("--out", default="data/head.npz")
    parser.add_argument("--model", default="allenai/OLMo-2-1124-7B-Instruct")
    parser.add_argument("--attack-ratio", type=float, default=0.5, help="attacks as a fraction of real rows")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    real = np.load(args.hidden, allow_pickle=False)
    hack = np.load(args.hack, allow_pickle=False) if glob.glob(args.hack) else None
    layers = [int(x) for x in real["layers"]]
    ids = [str(x) for x in real["ids"]]
    by_unit = load_labels(sorted(glob.glob(args.labels)))

    out: dict[str, np.ndarray] = {}
    meta = {"model": args.model, "dimensions": {}}
    for dim in DIMENSIONS:
        pooling, layer = CELLS[dim.key]
        li = layers.index(layer)
        rows, y = [], []
        for i, uid in enumerate(ids):
            scores = [r[dim.key] for r in by_unit.get(uid, {}).values() if isinstance(r.get(dim.key), int)]
            if scores:
                rows.append(i)
                y.append(statistics.fmean(scores))
        X = real[pooling][rows, li, :].astype(np.float32)
        target = np.array(y, dtype=np.float32)

        augmented = 0
        if dim.key == "leak" and hack is not None:
            rows_a = choose_attacks(hack, len(y), args.attack_ratio, args.seed)
            X = np.vstack([X, hack[pooling][rows_a, li, :].astype(np.float32)])
            augmented = len(rows_a)
            target = np.concatenate([target, np.full(augmented, float(dim.hi))])

        scaler = StandardScaler().fit(X)
        model = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(scaler.transform(X), target)
        out[f"{dim.key}/mean"] = scaler.mean_.astype(np.float32)
        out[f"{dim.key}/scale"] = scaler.scale_.astype(np.float32)
        out[f"{dim.key}/coef"] = model.coef_.astype(np.float32)
        out[f"{dim.key}/intercept"] = np.float32(model.intercept_)
        meta["dimensions"][dim.key] = {
            "pooling": pooling,
            "layer": layer,
            "lo": dim.lo,
            "hi": dim.hi,
            "n": len(rows),
            "augmented": augmented,
            "alpha": float(model.alpha_),
            "surface_baseline": SURFACE_BOUND.get(dim.key),
        }
        # A head whose output barely moves across real turns is a constant reward,
        # and it looks like a working head everywhere except in the spread. The
        # first fit here pinned every leak prediction at 3 and the failure surfaced
        # two steps later as a correlation of nan.
        on_real = np.clip(
            (real[pooling][rows, li, :].astype(np.float32) - scaler.mean_) / scaler.scale_ @ model.coef_
            + model.intercept_,
            dim.lo,
            dim.hi,
        )
        spread = float(on_real.std())
        if spread < 0.15:
            raise SystemExit(
                f"{dim.key}: predictions on real turns have std {spread:.3f} over {dim.lo}-{dim.hi}. "
                "That is a constant, not a reward. Lower --attack-ratio or check the labels."
            )

        # The attacks not used in fitting, scored by the head that is about to ship.
        # Balancing the augmentation traded some of them away, and this says how many.
        held = ""
        if dim.key == "leak" and hack is not None:
            rest = [i for i in range(hack[pooling].shape[0]) if i not in set(rows_a)]
            pred = (hack[pooling][rest, li, :].astype(np.float32) - scaler.mean_) / scaler.scale_ @ model.coef_
            missed = float((pred + model.intercept_ < 2.0).mean())
            held = f"  missed {missed:.0%} of {len(rest)} unseen attacks"
            if missed > 0.15:
                raise SystemExit(
                    f"leak: the shipping head misses {missed:.0%} of attacks it did not train on. "
                    "Raise --attack-ratio; a policy will find this."
                )

        note = f", +{augmented} attacks" if augmented else ""
        warn = "  SURFACE - a word counter, do not reward on this" if dim.key in SURFACE_BOUND else ""
        print(
            f"  {dim.key:<12} {pooling:>5} layer {layer:<3} n={len(rows)}{note}  "
            f"alpha={model.alpha_:g}  spread={spread:.2f}{held}{warn}"
        )

    np.savez_compressed(args.out, meta=np.array(json.dumps(meta)), **out)
    print(f"\nwrote {args.out}: {len(DIMENSIONS)} heads over {real[CELLS['leak'][0]].shape[-1]} dims")


if __name__ == "__main__":
    main()
