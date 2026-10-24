"""Summarize the observational chat-row substitutions, never qualify production."""

import argparse
import json
from pathlib import Path

from analyze_update_audits import load_records, vector_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root
    monitor = root / "chat-row-cf-cp8-monitor"
    receipt = json.loads((monitor / "receipt.json").read_text())
    assert receipt["passed"] and receipt["succeeded_pods"] == 32 and receipt["matched_hashes"] == 16
    records = []
    for path in sorted((monitor / "logs").glob("*.log")):
        for line in path.read_text().splitlines():
            if "H127-CHAT-ROW-CF " in line:
                records.append(json.loads(line.split("H127-CHAT-ROW-CF ", 1)[1]))
    assert records and all(r == records[0] for r in records)
    record = records[0]
    assert record["step"] == 2
    rows = {r["mode"]: r for r in record["rows"]}
    assert set(rows) == {"unchanged", "im_start", "im_end", "both"} and len(record["rows"]) == 4
    original_cp = json.loads((root / "gradient-cp8-monitor/receipt.json").read_text())
    original_ep = json.loads((root / "gradient-ep8-monitor/receipt.json").read_text())
    ep_ce, cp_ce = (r["loss_parts"]["2"]["ce"] for r in (original_ep, original_cp))
    assert all(row["tokens"] == original_cp["loss_parts"]["2"]["tokens"] for row in rows.values())
    baseline = rows["unchanged"]["ce"]
    for row in rows.values():
        row["ce_change_from_same_run_unchanged"] = row["ce"] - baseline
        row["remaining_gap_to_original_ep8"] = row["ce"] - ep_ce
    prior_audits = load_records(root / "gradient-cp8-monitor/logs")
    audits = load_records(monitor / "logs")
    assert set(audits) == {1, 2}
    assert audits[1]["token_ids"] == prior_audits[1]["token_ids"]
    comparisons = {}
    for key, array in audits[1]["arrays"].items():
        if key.startswith(("rows_before/", "row_gradients/", "rows_after/")):
            comparisons[key] = vector_comparison(array, prior_audits[1]["arrays"][key])
    result = {
        "scope": "Diagnostic causal row substitutions only; original numerical failure and qualification requirements unchanged",
        "ordinary_losses": receipt["loss_parts"],
        "interpretation_limit": "Compare substitutions within the extra-forward path. If its unchanged baseline differs from ordinary training, these values do not quantify mediation of the original cross-mesh gap.",
        "original_cp8_step2_ce": cp_ce,
        "original_ep8_step2_ce": ep_ce,
        "unchanged_forward_minus_ordinary_ce": baseline - receipt["loss_parts"]["2"]["ce"],
        "unchanged_forward_minus_original_cp8_ce": baseline - cp_ce,
        "step1_capture_comparison_with_original_cp8": comparisons,
        "rows": list(rows.values()),
    }
    path = root / "numerical-investigation/chat-row-counterfactual-result.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "step1_capture_comparison_with_original_cp8"}, indent=2))


if __name__ == "__main__":
    main()
