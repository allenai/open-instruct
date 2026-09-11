"""Prepare an isolated light-SFT retry with fully verified node-local HF bytes."""

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

from scripts.miles import light_sft_gsm8k
from scripts.miles.prepare_gsm8k_parity import json_bytes


def stage_hf(source, destination, *, timeout_seconds=1200):
    source, destination = Path(source), Path(destination)
    files = sorted(path for path in source.iterdir() if path.is_file())
    if not files or not any(path.suffix == ".safetensors" for path in files):
        raise ValueError("Staging requires an HF safetensors checkpoint")
    required = sum(path.stat().st_size for path in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < required + 1024**3:
        raise ValueError("Insufficient node-local space for the complete checkpoint")
    destination.mkdir()
    started = time.monotonic()
    report = {"source": str(source), "destination": str(destination), "files": {}}
    for path in files:
        before = path.stat()
        target = destination / path.name
        original = hashlib.sha256()
        copied = 0
        with path.open("rb") as reader, target.open("xb") as writer:
            while block := reader.read(16 * 1024**2):
                if time.monotonic() - started > timeout_seconds:
                    raise TimeoutError("Checkpoint staging exceeded its time budget")
                writer.write(block)
                original.update(block)
                copied += len(block)
                if copied % (1024**3) == 0:
                    print("HF_STAGING_PROGRESS", path.name, copied, flush=True)
            writer.flush()
            os.fsync(writer.fileno())
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError("Source checkpoint changed during staging")
        with target.open("rb") as reader:
            observed = hashlib.file_digest(reader, "sha256").hexdigest()
        if observed != original.hexdigest() or copied != before.st_size:
            raise ValueError("Staged checkpoint differs from the complete source byte stream")
        report["files"][path.name] = {"bytes": copied, "sha256": observed}
    report["elapsed_seconds"] = time.monotonic() - started
    report["verified_full_payload"] = True
    return report


def prepare_retry(source_root, root, local_hf):
    if root.exists():
        if (root / "core").exists():
            raise ValueError("Refusing to reuse an existing training attempt")
        # An explicitly selected new preparation may carry corrected proofs.
        # Verify it in place; never replace it with a prior attempt's proofs.
        prepared_source = root
    else:
        light_sft_gsm8k.verify(source_root)
        root.mkdir()
        for path in source_root.iterdir():
            if path.name != "core":
                (root / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
        prepared_source = source_root
    light_sft_gsm8k.verify(root)
    report = stage_hf(root / "hf", local_hf)
    report["prepared_source_root"] = str(prepared_source)
    report["retry_root"] = str(root)
    (root / "local-staging.json").write_bytes(json_bytes(report))
    print("LIGHT_SFT_STAGED", json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--local-hf", type=Path, required=True)
    args = parser.parse_args()
    prepare_retry(args.source_root, args.root, args.local_hf)


if __name__ == "__main__":
    main()
