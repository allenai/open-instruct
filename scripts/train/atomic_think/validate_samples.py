"""Verify local sample staging and reproduce the stopped control's batch audits."""

import argparse
import json
import time
from pathlib import Path

import torch
from scripts.train.atomic_think import prepare_samples

from open_instruct import atomic_think, packed_sample_cache


def verify_batches(loader, audit):
    checked = {}
    for rank in range(16):
        loader.dp_rank = rank
        for step in (*range(1, 13), 100):
            path = audit / f"batch-step{step}-rank{rank}.json"
            expected = json.loads(path.read_text())
            batch = loader[step - 1]
            actual = {
                key: atomic_think.tensor_hash(value) for key, value in batch.items() if isinstance(value, torch.Tensor)
            }
            if actual != expected:
                raise ValueError(f"Cached batch differs from original training: {path.name}")
            checked[path.name] = actual
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("kind", choices=("original", "atomic"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    started = time.monotonic()
    destination = Path("/tmp/h077-validated-samples")
    receipt = packed_sample_cache.stage_cache(args.root / f"sample-cache-{args.kind}", destination)
    receipt["staging_seconds"] = time.monotonic() - started
    dataset, loader, indices = prepare_samples.build_source(args.root, args.kind)
    cached = packed_sample_cache.CachedPackedFSLDataset(dataset, destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    if [entry["index"] for entry in manifest["rows"]] != indices[: len(manifest["rows"])].tolist():
        raise ValueError("Cached prefix differs from native shuffled order")
    started = time.monotonic()
    for entry in manifest["rows"]:
        cached[entry["index"]]
    receipt["cached_read_seconds"] = time.monotonic() - started
    receipt["verified_samples"] = len(manifest["rows"])
    loader.dataset = cached
    loader._global_indices = indices
    if args.kind == "original":
        receipt["training_batches"] = verify_batches(loader, args.root / "control" / "atomic-audit")
    atomic_think.write_json(Path("/output") / f"sample-validation-{args.kind}.json", receipt)


if __name__ == "__main__":
    main()
