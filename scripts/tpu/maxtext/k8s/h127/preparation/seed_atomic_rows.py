"""Verify native extraction and seed only the two atomic rows, preserving FP32 masters."""

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROWS = [(100256, [14023, 771, 29]), (100270, [524, 27963, 29])]
MATRICES = {"embeddings.weight.npy", "lm_head.w_out.weight.npy"}


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source, output = map(Path, sys.argv[1:])
    receipt = json.loads((source / "masters-receipt.json").read_text())
    assert sha256(source / "config.json") == receipt["config_sha256"]
    expected = {item["name"] for item in receipt["files"]}
    assert len(expected) == 415
    assert expected == {p.name for p in (source / "masters").glob("*.npy")}
    output.mkdir(exist_ok=False)
    (output / "masters").mkdir()
    shutil.copy2(source / "config.json", output / "config.json")
    changed = {}
    for item in receipt["files"]:
        path = source / "masters" / item["name"]
        assert path.stat().st_size == item["size"]
        assert sha256(path) == item["sha256"], path.name
        target = output / "masters" / path.name
        if path.name not in MATRICES:
            os.link(path, target)
            continue
        # Copy these two files: never modify a hardlink to the unmodified extraction.
        shutil.copy2(path, target)
        original = np.load(path, mmap_mode="r").reshape(100352, 1024)
        seeded = np.load(target, mmap_mode="r+").reshape(100352, 1024)
        for row, pieces in ROWS:
            tensor = torch.from_numpy(original[pieces].copy()).to(torch.bfloat16)
            seeded[row] = tensor.float().mean(0).to(torch.bfloat16).float().numpy()
        seeded.flush()
        differences = np.flatnonzero(np.any(seeded.view(np.uint32) != original.view(np.uint32), axis=1))
        assert set(differences.tolist()) <= {row for row, _ in ROWS}
        changed[path.name] = {
            "changed_rows": differences.tolist(),
            "file_sha256": sha256(target),
            "initial_rows_sha256": hashlib.sha256(seeded[[r for r, _ in ROWS]].tobytes()).hexdigest(),
        }
    result = {
        "source": receipt["source"],
        "source_config_sha256": receipt["config_sha256"],
        "verified_files": len(receipt["files"]),
        "initialization": "Mean of historical constituent BF16 rows, reduced in FP32, rounded to BF16; only target rows replaced in FP32 masters",
        "rows": ROWS,
        "changed": changed,
        "all_other_master_values_bitwise_preserved": True,
    }
    (output / "initialization-receipt.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
