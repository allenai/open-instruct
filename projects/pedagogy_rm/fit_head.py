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
            X = np.vstack([X, hack[pooling][:, li, :].astype(np.float32)])
            augmented = hack[pooling].shape[0]
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
        note = f", +{augmented} attacks" if augmented else ""
        warn = "  SURFACE - a word counter, do not reward on this" if dim.key in SURFACE_BOUND else ""
        print(f"  {dim.key:<12} {pooling:>5} layer {layer:<3} n={len(rows)}{note}  alpha={model.alpha_:g}{warn}")

    np.savez_compressed(args.out, meta=np.array(json.dumps(meta)), **out)
    print(f"\nwrote {args.out}: {len(DIMENSIONS)} heads over {real[CELLS['leak'][0]].shape[-1]} dims")


if __name__ == "__main__":
    main()
