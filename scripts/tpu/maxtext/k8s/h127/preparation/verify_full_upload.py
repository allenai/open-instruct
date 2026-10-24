"""Verify every uploaded full-corpus object against local SHA256 and GCS CRC32C."""

import base64
import concurrent.futures
import datetime
import hashlib
import json
import subprocess
import time
from pathlib import Path

import google_crc32c

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/robert/storage/dolci-think-sft/tpu-presentation14t-full-v1")
DEST = "gs://ai2-olmo-dev/post-training/data/robertb/tpu-presentation14t-full-v1/"
GCLOUD = "/snap/google-cloud-cli/current/bin/gcloud"
manifest = json.loads((DATA / "manifest.json").read_text())


def inspect(path):
    relative = str(path.relative_to(DATA))
    sha, crc = hashlib.sha256(), google_crc32c.Checksum()
    before = path.stat()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            sha.update(chunk)
            crc.update(chunk)
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), relative
    result = {"size": after.st_size, "sha256": sha.hexdigest(), "crc32c": base64.b64encode(crc.digest()).decode()}
    if relative in manifest["files"]:
        expected = manifest["files"][relative]
        assert result["sha256"] == expected["sha256"] and result["size"] == expected["bytes"], relative
    return relative, result


paths = sorted(p for p in DATA.rglob("*") if p.is_file())
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    local = dict(pool.map(inspect, paths))
assert len(local) == 1029 and set(manifest["files"]) <= set(local)
(ROOT / "full-data-integrity-local.json").write_text(json.dumps(local, indent=2) + "\n")
print("LOCAL-INTEGRITY", len(local), sum(r["size"] for r in local.values()), flush=True)
# Readiness means the existing rsync exited successfully. This watcher does not
# modify or restart that process. An observation deadline leaves it alone.
started = time.monotonic()
while not (ROOT / "full-data-ready.json").exists():
    if time.monotonic() - started > 8 * 3600:
        raise TimeoutError("Upload not yet ready; inspect existing upload, do not restart it")
    time.sleep(30)
raw = subprocess.check_output([GCLOUD, "storage", "ls", "--json", DEST + "**"], text=True, timeout=180)
(ROOT / "full-data-gcs-objects.json").write_text(raw)
remote = {}
for entry in json.loads(raw):
    assert entry["type"] == "cloud_object"
    m = entry["metadata"]
    url = "gs://" + m["bucket"] + "/" + m["name"]
    assert url.startswith(DEST)
    remote[url[len(DEST) :]] = {"size": int(m["size"]), "crc32c": m["crc32c"], "generation": m["generation"]}
assert set(remote) == set(local), {
    "missing": sorted(set(local) - set(remote)),
    "extra": sorted(set(remote) - set(local)),
}
for name, expected in local.items():
    assert all(remote[name][key] == expected[key] for key in ("size", "crc32c")), name
receipt = {
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "gcs_prefix": DEST,
    "objects": len(local),
    "bytes": sum(r["size"] for r in local.values()),
    "all_sizes_and_crc32c_match": True,
    "all_1024_parquet_sha256_match_prepared_manifest": True,
    "local_integrity_sha256": hashlib.sha256((ROOT / "full-data-integrity-local.json").read_bytes()).hexdigest(),
    "remote_listing_sha256": hashlib.sha256(raw.encode()).hexdigest(),
}
(ROOT / "full-data-integrity-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("GCS-INTEGRITY-PASSED", json.dumps(receipt), flush=True)
