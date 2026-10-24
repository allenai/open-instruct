"""Protect paired identity, checkpoint alignment and sign of the primary read."""

import copy
import json
import tempfile
from pathlib import Path

from compare_openers import compare, load

control = {
    ("original", str(i)): {
        "qualified_id": str(i),
        "variant": "original",
        "input_ids_sha256": str(i),
        "step": 2048,
        "newline_family": 0.2,
        "Okay": 0.3,
        "space_Okay": 0.1,
        "remainder": 0.4,
        "token_probabilities": {"198": 0.15, "271": 0.05},
    }
    for i in range(8)
}
treatment = copy.deepcopy(control)
for r in treatment.values():
    r["newline_family"] -= 0.1
    r["remainder"] += 0.1
    r["token_probabilities"] = {"198": 0.08, "271": 0.02}
expected = compare(control, treatment, n_items=8)
metric = expected["rows"][0]["metrics"]["newline_family"]
assert abs(metric["delta"] + 0.1) < 1e-12
assert all(abs(x + 0.1) < 1e-12 for x in metric["ci95"])
assert compare(dict(reversed(list(control.items()))), treatment, n_items=8) == expected
for kind in ("missing", "prompt", "step"):
    bad = copy.deepcopy(treatment)
    key = next(iter(bad))
    if kind == "missing":
        del bad[key]
    elif kind == "prompt":
        bad[key]["input_ids_sha256"] = "wrong"
    else:
        bad[key]["step"] = 512
    try:
        compare(control, bad, n_items=8)
    except AssertionError:
        pass
    else:
        raise AssertionError(f"Accepted {kind} mismatch")
with tempfile.TemporaryDirectory() as folder:
    p = Path(folder) / "records.jsonl"
    rows = list(control.values())
    p.write_text("\n".join(json.dumps(r) for r in rows))
    assert load(p) == control
    p.write_text("\n".join(json.dumps(r) for r in rows + rows[:1]))
    try:
        load(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate accepted")
print("OPENER-COMPARISON-TEST-PASSED")
