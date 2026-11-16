"""Install the experimental denominator into exactly the H235 native runtime."""

import hashlib
import json
import os
import shutil
from pathlib import Path

NATIVE_PATH = Path("/opt/core-rl/sources/miles/miles/ray/rollout/train_data_conversion.py")
OLD = "rollout_std = rollout_rewards.std()"
NEW = "rollout_std = bernoulli.reference_std(rollout_rewards)"
IMPORT = "from open_instruct.miles.rewards import bernoulli\n"


def patched_source(original: str, expected_hash: str) -> str:
    if hashlib.sha256(original.encode()).hexdigest() != expected_hash:
        raise ValueError("native reward conversion differs from pinned H235 source")
    if original.count(OLD) != 1:
        raise ValueError("expected exactly one native standard deviation expression")
    return original.replace("import torch\n", "import torch\n\n" + IMPORT, 1).replace(OLD, NEW, 1)


def main():
    source = Path(os.environ["OI_BERNOULLI_SOURCE"])
    manifest_bytes = (source / "MANIFEST.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != os.environ["OI_BERNOULLI_MANIFEST_SHA256"]:
        raise ValueError("binary-reference source manifest mismatch")
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"binary-reference source mismatch: {name}")
    original = (source / "native.py").read_text()
    patched = patched_source(original, manifest["native_sha256"])
    if NATIVE_PATH.read_text() not in (original, patched):
        raise ValueError("refusing to overwrite an unexpected native runtime")
    destination = Path("/opt/core-rl/open_instruct/miles/rewards/bernoulli.py")
    shutil.copyfile(source / "bernoulli.py", destination)
    scripts = Path("/opt/core-rl/scripts/miles/bernoulli")
    scripts.mkdir(parents=True, exist_ok=True)
    for name in ("install.py", "test_bernoulli.py"):
        shutil.copyfile(source / name, scripts / name)
    compile(patched, str(NATIVE_PATH), "exec")
    NATIVE_PATH.write_text(patched)
    print(
        "H235 binary-reference normalization installed; manifest=" + hashlib.sha256(manifest_bytes).hexdigest(),
        flush=True,
    )


if __name__ == "__main__":
    main()
