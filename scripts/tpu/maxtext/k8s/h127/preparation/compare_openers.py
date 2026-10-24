"""H127 paired question-bootstrap read; never substitutes partial panels or seeds."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

GROUPS = ("newline_family", "Okay", "space_Okay", "remainder")


def load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        rows = [json.loads(line) for line in stream]
    keyed = {(r.get("variant", "original"), r["qualified_id"]): r for r in rows}
    assert len(keyed) == len(rows), "Duplicate question/variant"
    for r in rows:
        values = [r[g] for g in GROUPS]
        assert all(np.isfinite(v) and -1e-6 <= v <= 1 + 1e-6 for v in values)
        assert abs(sum(values) - 1) < 1e-5
    return keyed


def compare(control, treatment, n_items=1000):
    assert set(control) == set(treatment), "Missing/unmatched questions or variants"
    variants = sorted({k[0] for k in control})
    output = []
    for variant in variants:
        keys = sorted(k for k in control if k[0] == variant)
        assert len(keys) == n_items
        assert len({control[k]["step"] for k in keys}) == 1
        for key in keys:
            a, b = control[key], treatment[key]
            assert a["input_ids_sha256"] == b["input_ids_sha256"], "Prompt mismatch"
            assert a["step"] == b["step"], "Checkpoint mismatch"
            assert a.get("split") == b.get("split")
        names = list(GROUPS) + ["P198", "P271"]

        def vector(r):
            return [r[g] for g in GROUPS] + [r["token_probabilities"][t] for t in ("198", "271")]

        original = np.asarray([vector(control[k]) for k in keys], dtype=np.float64)
        changed = np.asarray([vector(treatment[k]) for k in keys], dtype=np.float64)
        delta = changed - original
        rng = np.random.default_rng(61023)
        bootstrap = np.concatenate(
            [delta[rng.integers(0, len(keys), size=(100, len(keys)))].mean(axis=1) for _ in range(100)]
        )
        low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        output.append(
            {
                "variant": variant,
                "step": control[keys[0]]["step"],
                "n": len(keys),
                "metrics": {
                    name: {
                        "control": float(original[:, i].mean()),
                        "treatment": float(changed[:, i].mean()),
                        "delta": float(delta[:, i].mean()),
                        "ci95": [float(low[i]), float(high[i])],
                    }
                    for i, name in enumerate(names)
                },
            }
        )
    return {
        "method": "Paired question bootstrap, 10000 resamples, seed 61023; treatment minus control",
        "rows": output,
        "limitations": "Exploratory, one training seed per arm. Intervals cover question variation, not training-seed variation; secondary contrasts are unadjusted. No adoption or no-regression claim.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    control, treatment = load(args.control), load(args.treatment)
    assert {r.get("arm") for r in control.values()} == {"atomic-original"}
    assert {r.get("arm") for r in treatment.values()} == {"atomic-presentation-balanced"}
    args.output.write_text(json.dumps(compare(control, treatment), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
