"""Reject partial, mixed, modified or wrong-prompt secondary generation records."""

import copy
import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from score_math import validate_generations

sampling = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": -1,
    "max_new_tokens": 32768,
    "stop_token_ids": [100257, 100265],
    "skip_special_tokens": True,
    "no_stop_trim": False,
}
record = {
    "qualified_id": "q",
    "fingerprint": "f",
    "sampling": sampling,
    "arm": "atomic-original",
    "step": 2048,
    "input_ids": [1, 2, 100256],
    "body_ids": [33413],
    "finish_reason": {"type": "stop"},
}
manifest = {"pilot_items": None, "fingerprint": "f", "arm": "atomic-original", "step": 2048, "sampling": sampling}
with tempfile.TemporaryDirectory() as directory:
    folder = Path(directory)
    data = folder / "generation-q.json.gz"

    def save(r, m):
        data.write_bytes(gzip.compress(json.dumps(r).encode(), mtime=0))
        m = copy.deepcopy(m)
        m["rows"] = [{"qualified_id": "q", "file": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()}]
        (folder / "generation-complete.json").write_text(json.dumps(m))

    save(record, manifest)
    panel = {"q": {"input_ids": [1, 2, 100256]}}
    assert validate_generations([folder], {"q": {}}, panel)[0]["q"] == record
    for case in ("pilot", "prompt", "step", "sampling", "missing", "hash"):
        r, m = copy.deepcopy(record), copy.deepcopy(manifest)
        if case == "pilot":
            m["pilot_items"] = 1
        if case == "prompt":
            r["input_ids"] = [3, 2, 100256]
        if case == "step":
            r["step"] = 512
        if case == "sampling":
            r["sampling"]["temperature"] = 0
        save(r, m)
        if case == "hash":
            data.write_bytes(data.read_bytes() + b"modified")
        requests = {"q": {}, "missing": {}} if case == "missing" else {"q": {}}
        try:
            validate_generations([folder], requests, panel)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Accepted {case}")
print("MATH-RECORD-TEST-PASSED")
