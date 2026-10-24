"""Decode H127 update diagnostics and compare gradients; no qualification verdict."""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def load_records(directory):
    records = {}
    for path in sorted(directory.glob("*.log")):
        for line in path.read_text().splitlines():
            if "H127-UPDATE-AUDIT " not in line:
                continue
            record = json.loads(line.split("H127-UPDATE-AUDIT ", 1)[1])
            raw = base64.b64decode(record["npz_base64"], validate=True)
            assert hashlib.sha256(raw).hexdigest() == record["sha256"]
            with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
                arrays = {k: np.asarray(archive[k], dtype=np.float64) for k in archive.files}
            assert len(arrays) == record["array_count"]
            assert all(np.isfinite(v).all() for v in arrays.values())
            step = record["step"]
            assert step not in records or records[step]["sha256"] == record["sha256"]
            records[step] = {**record, "arrays": arrays}
    return records


def vector_comparison(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {
        "candidate_norm": na,
        "reference_norm": nb,
        "cosine": float(np.vdot(a, b) / (na * nb)) if na and nb else None,
        "difference_norm": float(np.linalg.norm(a - b)),
        "relative_difference": float(np.linalg.norm(a - b) / nb) if nb else None,
        "exactly_equal": bool(np.array_equal(a, b)),
    }


def summarize(record):
    arrays = record["arrays"]
    norms = {k.removeprefix("gradient_norm2/"): float(v) for k, v in arrays.items() if k.startswith("gradient_norm2/")}
    total = sum(norms.values())
    assert total > 0 and all(v >= 0 for v in norms.values())
    tables = {}
    for key, grad in arrays.items():
        if not key.startswith("row_gradients/"):
            continue
        name = key.removeprefix("row_gradients/")
        before, after = arrays["rows_before/" + name], arrays["rows_after/" + name]
        assert grad.shape == before.shape == after.shape and len(grad) == len(record["token_ids"])
        tables[name] = [
            {
                "id": token,
                "gradient_norm": float(np.linalg.norm(g)),
                "gradient_squared_share_of_total": float(np.vdot(g, g) / total),
                "before_norm": float(np.linalg.norm(b)),
                "after_norm": float(np.linalg.norm(a)),
                "update_norm": float(np.linalg.norm(a - b)),
            }
            for token, g, b, a in zip(record["token_ids"], grad, before, after, strict=True)
        ]
    return {
        "gradient_norm": total**0.5,
        "parameter_leaves": len(norms),
        "largest_gradient_contributors": [
            {"name": k, "norm": v**0.5, "squared_share": v / total}
            for k, v in sorted(norms.items(), key=lambda x: x[1], reverse=True)[:12]
        ],
        "selected_token_rows": tables,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate_logs", type=Path)
    parser.add_argument("reference_logs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, choices=(3, 6), default=3)
    args = parser.parse_args()
    arms = [load_records(p) for p in [args.candidate_logs, args.reference_logs]]
    expected_steps = set(range(1, args.steps + 1))
    assert all(set(a) == expected_steps for a in arms), f"Require all {args.steps} steps from each mesh"
    result = {"scope": "Diagnostic attribution only; does not override the failed numerical gate", "steps": []}
    for step in range(1, args.steps + 1):
        c, r = (a[step] for a in arms)
        assert c["token_ids"] == r["token_ids"] and c["arrays"].keys() == r["arrays"].keys()
        comparison = {}
        for key in c["arrays"]:
            if not key.startswith("row_gradients/"):
                continue
            name = key.removeprefix("row_gradients/")
            rows = []
            for i, token in enumerate(c["token_ids"]):
                ca, ra = c["arrays"], r["arrays"]
                rows.append(
                    {
                        "id": token,
                        "before": vector_comparison(ca["rows_before/" + name][i], ra["rows_before/" + name][i]),
                        "gradient": vector_comparison(ca[key][i], ra[key][i]),
                        "update": vector_comparison(
                            ca["rows_after/" + name][i] - ca["rows_before/" + name][i],
                            ra["rows_after/" + name][i] - ra["rows_before/" + name][i],
                        ),
                    }
                )
            comparison[name] = rows
        result["steps"].append(
            {"step": step, "candidate": summarize(c), "reference": summarize(r), "row_comparison": comparison}
        )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"steps": len(result["steps"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
