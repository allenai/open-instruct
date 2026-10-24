"""Wait for the original uploader to exit; conditionally create missing objects only."""

import collections
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/robert/storage/dolci-think-sft/tpu-presentation14t-full-v1")
DEST = "gs://ai2-olmo-dev/post-training/data/robertb/tpu-presentation14t-full-v1/"
GCLOUD = "/snap/google-cloud-cli/current/bin/gcloud"
assert Path("/proc/1/comm").read_text().strip() == "systemd", "Must inspect host PIDs, not sandbox namespace"


def live(pid, needle):
    proc = Path("/proc") / str(pid)
    if not proc.exists():
        return False
    try:
        state = (proc / "stat").read_text().split(") ", 1)[1].split()[0]
        if state == "Z":
            return False
        cmd = (proc / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    assert needle.encode() in cmd, f"PID {pid} changed identity; inspect before proceeding"
    return True


def inventory(label):
    raw = subprocess.check_output([GCLOUD, "storage", "ls", "--json", DEST + "**"], text=True, timeout=180)
    (ROOT / f"full-data-repair-{label}-objects.json").write_text(raw)
    result = {}
    for item in json.loads(raw):
        assert item["type"] == "cloud_object"
        meta = item["metadata"]
        url = "gs://" + meta["bucket"] + "/" + meta["name"]
        assert url.startswith(DEST)
        result[url[len(DEST) :]] = {"size": int(meta["size"]), "crc32c": meta["crc32c"]}
    return result


started = time.monotonic()
while live(112304, "upload_full_when_ready.py") or live(185478, "rsync"):
    if time.monotonic() - started > 6 * 3600:
        raise TimeoutError("Observation deadline only; original uploader left untouched")
    time.sleep(30)
print("ORIGINAL-UPLOADER-TERMINAL", flush=True)
local = json.loads((ROOT / "full-data-integrity-local.json").read_text())
assert len(local) == 1029
remote = inventory("before")
assert set(remote) <= set(local), "Unexpected objects in immutable prepared-data prefix"
for name, actual in remote.items():
    assert actual == {k: local[name][k] for k in ("size", "crc32c")}, f"Existing object mismatch: {name}"
missing = sorted(set(local) - set(remote))
(ROOT / "full-data-repair-missing.json").write_text(json.dumps(missing, indent=2) + "\n")
print("CONFIRMED-MISSING-OBJECTS", len(missing), flush=True)
groups = collections.defaultdict(list)
for name in missing:
    path = Path(name)
    assert not path.is_absolute() and ".." not in path.parts
    source = DATA / path
    assert source.stat().st_size == local[name]["size"]
    with source.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == local[name]["sha256"], name
    groups[str(path.parent)].append(str(source))
env = dict(os.environ, CLOUDSDK_STORAGE_PROCESS_COUNT="2", CLOUDSDK_STORAGE_THREAD_COUNT="4")
for index, (parent, sources) in enumerate(sorted(groups.items())):
    destination = DEST + ("" if parent == "." else parent + "/")
    subprocess.run(
        [
            GCLOUD,
            "storage",
            "cp",
            "--no-clobber",
            "--if-generation-match=0",
            "--manifest-path=" + str(ROOT / f"full-data-repair-group{index}.csv"),
            "--read-paths-from-stdin",
            destination,
        ],
        input="\n".join(sources) + "\n",
        text=True,
        env=env,
        check=True,
    )
remote = inventory("after")
assert set(remote) == set(local), sorted(set(local) - set(remote))
for name, actual in remote.items():
    assert actual == {k: local[name][k] for k in ("size", "crc32c")}, name
receipt = {
    "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "original_uploader_terminal_before_copy": True,
    "uploaded_missing_objects": len(missing),
    "overwrite_precondition": "ifGenerationMatch=0, plus no-clobber",
    "all_1029_objects_sizes_crc32c_match": True,
    "bytes": sum(v["size"] for v in local.values()),
}
(ROOT / "full-data-repair-verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
if not (ROOT / "full-data-ready.json").exists():
    (ROOT / "full-data-ready.json").write_text(
        json.dumps({"gcs_prefix": DEST, "source": "verified missing-file repair", **receipt}, indent=2) + "\n"
    )
print("REPAIRED-UPLOAD-VERIFIED", json.dumps(receipt), flush=True)
# Preserve the existing verifier if it is still alive; otherwise rerun it after
# the original process is terminal. Never interpret an observation timeout as failure.
started = time.monotonic()
while not (ROOT / "full-data-integrity-receipt.json").exists():
    if not live(285152, "verify_full_upload.py"):
        subprocess.run([sys.executable, str(ROOT / "verify_full_upload.py")], check=True)
        break
    if time.monotonic() - started > 1800:
        raise TimeoutError("Existing verifier still live; inspect its process, do not restart it")
    time.sleep(30)
final = json.loads((ROOT / "full-data-integrity-receipt.json").read_text())
assert final["objects"] == 1029 and final["all_sizes_and_crc32c_match"]
print("FULL-DATA-QUALIFIED", flush=True)
