"""Copy a completed smoke checkpoint for a distinct, audited resume test."""

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path("/home/robert/proj/open-instruct/runs/presentation-tpu-h127-20261023")
GCLOUD = "/snap/google-cloud-cli/current/bin/gcloud"
SRC = "gs://ai2-olmo-dev/post-training/runs/robertb/presentation14t-v1/robertb-pres14t-v1-control-smoke-r2/checkpoints/50/"
DST = "gs://ai2-olmo-dev/post-training/runs/robertb/presentation14t-v1/robertb-pres14t-v1-control-resume50/checkpoints/50/"


def listing(prefix, empty_ok=False):
    result = subprocess.run(
        [GCLOUD, "storage", "ls", "--json", prefix + "**"], capture_output=True, text=True, timeout=180
    )
    if result.returncode:
        if empty_ok and "matched no objects" in result.stderr:
            return {}
        raise RuntimeError(result.stderr)
    rows = {}
    for entry in json.loads(result.stdout):
        assert entry["type"] == "cloud_object"
        m = entry["metadata"]
        url = "gs://" + m["bucket"] + "/" + m["name"]
        assert url.startswith(prefix)
        name = url[len(prefix) :]
        rows[name] = {"size": int(m["size"]), "crc32c": m["crc32c"], "generation": m["generation"]}
    return rows


receipt = json.loads((ROOT / "smoke-r2-monitor/receipt.json").read_text())
assert receipt["passed"] and receipt["last_step"] == 100 and receipt["succeeded_pods"] == 32
assert not receipt["failures"] and not receipt["missing"]
source = listing(SRC)
assert {"_CHECKPOINT_METADATA", "commit_success.txt"} <= set(source)
assert any(n.startswith("model_params/") for n in source)
assert any(n.startswith("optimizer_state/") for n in source)
assert not any("orbax-checkpoint-tmp" in n for n in source)
metadata = json.loads(subprocess.check_output([GCLOUD, "storage", "cat", SRC + "_CHECKPOINT_METADATA"], text=True))
assert metadata["commit_timestamp_nsecs"] > metadata["init_timestamp_nsecs"] > 0
assert {"model_params", "optimizer_state"} <= set(metadata["item_handlers"])
(ROOT / "resume-source-step50-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
existing = listing(DST, empty_ok=True)
for name, m in existing.items():
    assert name in source and all(m[k] == source[name][k] for k in ("size", "crc32c")), (
        "Unexpected existing destination object: " + name
    )
(ROOT / "resume-copy-source-manifest.json").write_text(json.dumps(source, indent=2) + "\n")
subprocess.run([GCLOUD, "storage", "rsync", "--recursive", "--no-clobber", SRC, DST], check=True)
target = listing(DST)
source_after = listing(SRC)
assert source_after == source, "Source checkpoint changed during copy"
assert set(source) == set(target)
assert all(all(source[n][k] == target[n][k] for k in ("size", "crc32c")) for n in source)
(ROOT / "resume-copy-target-manifest.json").write_text(json.dumps(target, indent=2) + "\n")
result = {
    "source": SRC,
    "destination": DST,
    "objects": len(source),
    "bytes": sum(m["size"] for m in source.values()),
    "all_names_sizes_crc32c_match": True,
    "source_generations_unchanged": True,
    "source_smoke_monitor_sha256": hashlib.sha256((ROOT / "smoke-r2-monitor/receipt.json").read_bytes()).hexdigest(),
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(ROOT / "resume-copy-receipt.json").write_text(json.dumps(result, indent=2) + "\n")
print("RESUME-COPY-VERIFIED", json.dumps(result), flush=True)
