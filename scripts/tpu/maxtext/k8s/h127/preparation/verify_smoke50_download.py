"""Verify the trained model download against the immutable GCS copy manifest."""

import base64
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path

import google_crc32c

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/robert/storage/presentation14t-smoke-r2-step50/model_params")
source = json.loads((ROOT / "resume-copy-source-manifest.json").read_text())
expected = {n.removeprefix("model_params/"): v for n, v in source.items() if n.startswith("model_params/")}
files = {str(f.relative_to(DATA)): f for f in DATA.rglob("*") if f.is_file()}
assert set(files) == set(expected), "Incomplete or unexpected parameter objects"


def verify(item):
    name, path = item
    crc = google_crc32c.Checksum()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            crc.update(chunk)
    assert path.stat().st_size == expected[name]["size"], name
    assert base64.b64encode(crc.digest()).decode() == expected[name]["crc32c"], name
    return path.stat().st_size


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    sizes = list(pool.map(verify, files.items()))
receipt = {
    "source": "gs://ai2-olmo-dev/post-training/runs/robertb/presentation14t-v1/robertb-pres14t-v1-control-smoke-r2/checkpoints/50/model_params",
    "local": str(DATA),
    "objects": len(files),
    "bytes": sum(sizes),
    "all_object_names_sizes_crc32c_match": True,
    "source_manifest_sha256": hashlib.sha256((ROOT / "resume-copy-source-manifest.json").read_bytes()).hexdigest(),
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
(ROOT / "smoke50-download-verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
