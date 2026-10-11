"""Download a released Olmo 3 7B Instruct stage at a pinned revision to WEKA, then read + sha256 every file once
(WEKA cold-tier trap; fresh writes land on SSD) and write a receipt. Run with HF_HOME=/opt/scratch/hf and
huggingface_hub==1.16.1 (the newest hub release failed mid-download on an httpx2 decoder error).
  python download_released.py [sft|dpo]
The DPO repo also ships pytorch_model-*.bin copies of the weights; only the safetensors are fetched."""

import fnmatch
import hashlib
import pathlib
import sys
import time

from huggingface_hub import HfApi, snapshot_download

STAGES = {
    "sft": ("allenai/Olmo-3-7B-Instruct-SFT", "e1452fc572d51966ff4aaeb25118b891eb93e549"),
    "dpo": ("allenai/Olmo-3-7B-Instruct-DPO", "b33130b7de49f0c2553b5c2b3bc8409ff3e627d1"),
}
IGNORE = ["pytorch_model*.bin", "pytorch_model.bin.index.json"]
stage = sys.argv[1] if len(sys.argv) > 1 else "sft"
REPO, REV = STAGES[stage]
DST = f"/weka/oe-adapt-default/abhishekr/tpu-posttrain/evals/models/olmo3-7b-instruct-{stage}-released"
t0 = time.time()
p = snapshot_download(REPO, revision=REV, local_dir=DST, ignore_patterns=IGNORE)
print("downloaded", p, f"{time.time() - t0:.0f}s", flush=True)
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
assert info.sha == REV, info.sha
lines = []
for s in info.siblings:
    if any(fnmatch.fnmatch(s.rfilename, pat) for pat in IGNORE):
        continue
    f = pathlib.Path(DST) / s.rfilename
    h = hashlib.sha256()
    n = 0
    with open(f, "rb") as fh:
        while chunk := fh.read(16 << 20):
            h.update(chunk)
            n += len(chunk)
    exp = s.lfs.sha256 if s.lfs else None
    assert n == s.size, (s.rfilename, n, s.size)
    if exp:
        assert h.hexdigest() == exp, (s.rfilename, h.hexdigest(), exp)
    lines.append(f"{h.hexdigest()}  {n}  {s.rfilename}" + ("  lfs-verified" if exp else ""))
(pathlib.Path(DST).parent / f"olmo3-7b-instruct-{stage}-released.receipt.txt").write_text(
    f"repo {REPO}\nrevision {REV}\nread+hashed every file after download, {time.strftime('%Y-%m-%d %H:%M:%S %Z')}\n"
    + "\n".join(lines)
    + "\n"
)
print("\n".join(lines))
print("total", f"{time.time() - t0:.0f}s")
