"""Summarize the registered six-step precision diagnostic; never qualify production."""

import argparse
import json
from pathlib import Path


def compare(candidate, reference):
    for r in (candidate, reference):
        assert r["succeeded_pods"] == 32 and r["failed_pods"] == 0
        assert r["matched_hashes"] == r["expected_hashes"] == 48
        assert not r["batch_hash_mismatches"] and not r["missing"]
        assert set(r["loss_parts"]) == {str(i) for i in range(1, 7)}
        assert set(r["drops"]) == {str(i) for i in range(1, 7)}
        assert all(d["dropped"] == 0 for d in r["drops"].values()), "Precision attribution requires zero drops"
    rows = []
    for step in range(1, 7):
        a, b = (r["loss_parts"][str(step)] for r in (candidate, reference))
        assert a["tokens"] == b["tokens"]
        assert a["moe_lb"] == b["moe_lb"] == 0.0
        rows.append({"step": step, "candidate_ce": a["ce"], "reference_ce": b["ce"], "gap": abs(a["ce"] - b["ce"])})
    gap = max(r["gap"] for r in rows)
    return {
        "scope": "Six-step FP32/highest precision, smaller output tiles and 2.4 expert buffers; original-data diagnostic only",
        "candidate": candidate["job"],
        "reference": reference["job"],
        "rows": rows,
        "maximum_ce_gap": gap,
        "existing_ce_bound": 0.001,
        "within_existing_ce_bound": gap <= 0.001,
        "individual_checks": {
            r["job"]: {
                "passed": r["passed"],
                "failures": r["failures"],
                "step1_gap_vs_bf16_hf_reference": r["step1_ce_gap"],
                "drop_fraction": r["drop_fraction"],
                "max_step_drop_fraction": r["max_step_drop_fraction"],
            }
            for r in (candidate, reference)
        },
        "production_qualified": False,
        "limitation": "Does not replace either failed BF16 comparison, a complete 20-step qualification, or treatment-data checks.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = compare(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
