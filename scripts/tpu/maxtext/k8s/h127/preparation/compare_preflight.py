"""Compare registered full-corpus candidate and CP8 reference receipts; no launches."""

import argparse
import json
from pathlib import Path


def compare(candidate, reference):
    assert candidate["passed"] and reference["passed"], "Both complete preflights must qualify individually"
    assert candidate["succeeded_pods"] == reference["succeeded_pods"] == 32
    for receipt in (candidate, reference):
        assert receipt["matched_hashes"] == receipt["expected_hashes"] == 160
        assert not receipt["batch_hash_mismatches"] and not receipt["failures"] and not receipt["missing"]
        assert set(receipt["loss_parts"]) == {str(i) for i in range(1, 21)}
        assert receipt["drop_fraction"] <= 1e-4 and receipt["max_step_drop_fraction"] <= 0.01
    rows = []
    for step in range(1, 21):
        a, b = (r["loss_parts"][str(step)] for r in (candidate, reference))
        assert a["tokens"] == b["tokens"], f"Different supervised exposure at step {step}"
        # Pre-registered preflights disable router LB and router Z on BOTH meshes.
        assert a["moe_lb"] == b["moe_lb"] == 0.0, "Unexpected auxiliary loss"
        rows.append(
            {
                "step": step,
                "tokens": a["tokens"],
                "candidate_ce": a["ce"],
                "reference_ce": b["ce"],
                "absolute_gap": abs(a["ce"] - b["ce"]),
            }
        )
    first_six = max(r["absolute_gap"] for r in rows[:6])
    assert first_six <= 1e-3, f"Cross-mesh numerical gate failed: {first_six}"
    return {
        "candidate": candidate["job"],
        "reference": reference["job"],
        "registered_max_gap": 1e-3,
        "first_six_max_gap": first_six,
        "rows": rows,
        "passed": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.candidate.read_text()), json.loads(args.reference.read_text()))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
