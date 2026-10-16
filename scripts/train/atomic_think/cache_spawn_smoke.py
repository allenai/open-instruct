"""Exercise the full registered cache through Core's spawned CPU data loader."""

import argparse
import gc
import hashlib
import multiprocessing
import pickle
import resource
import time
from pathlib import Path

import torch
from scripts.train.atomic_think import prepare_samples

from open_instruct import atomic_think, packed_sample_cache

MANIFESTS = {
    "original": "d9419ce76bd4c2b33ff08c1f40bc3a1ef5a40eda4b41c522bf63df2bade7a7ca",
    "atomic": "10aed9a83feae20b1179413b58c8bb84541689f5ee851402e5e376501c45a0f7",
}


def hashes(batch):
    return {key: atomic_think.tensor_hash(value) for key, value in batch.items() if isinstance(value, torch.Tensor)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("cache_root", type=Path)
    args = parser.parse_args()
    multiprocessing.set_start_method("spawn", force=True)
    torch.set_num_threads(1)
    reports = {}
    for kind, expected in MANIFESTS.items():
        directory = args.cache_root / f"full-sample-cache-{kind}"
        manifest = (directory / "manifest.json").read_bytes()
        if hashlib.sha256(manifest).hexdigest() != expected:
            raise ValueError("Wrong registered full cache")
        dataset, loader, _ = prepare_samples.build_source(args.data_root, kind)
        cached = packed_sample_cache.CachedPackedFSLDataset(dataset, directory)
        loader.dataset = cached
        loader.num_workers = 4
        payload = pickle.dumps(loader)
        mapped_bytes = cached._cached_tokens.nbytes + cached._cached_masks.nbytes
        if len(payload) >= mapped_bytes // 1000:
            raise ValueError("Worker startup payload is unexpectedly large")
        restored = pickle.loads(payload)
        if restored.dataset._cached_tokens.filename is None or restored.dataset._cached_masks.filename is None:
            raise ValueError("Spawn would lose file-backed cache storage")
        report = {
            "manifest_sha256": expected,
            "mapped_bytes": mapped_bytes,
            "loader_pickle_bytes": len(payload),
            "workers": 4,
            "ranks": {},
        }
        del payload, restored
        for rank in (0, 15):
            loader.dp_rank = rank
            loader.batches_processed = 512
            loader.tokens_processed = 512 * loader.global_batch_size
            expected_batches = [hashes(loader[step]) for step in range(512, 516)]
            started = time.monotonic()
            iterator = iter(loader)
            actual = [hashes(next(iterator)) for _ in range(4)]
            iterator.close()
            del iterator
            gc.collect()
            if actual != expected_batches or loader.batches_processed != 516:
                raise ValueError("Spawned loader changed the resumed data order or tensors")
            report["ranks"][str(rank)] = {"seconds": time.monotonic() - started, "batches": actual}
            print(f"Verified {kind} rank {rank}: four spawned workers, resumed batches 513-516", flush=True)
        reports[kind] = report
    reports["peak_rss_kib"] = {
        "parent": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "largest_child": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    atomic_think.write_json(Path("/output/cache-spawn-smoke.json"), reports)


if __name__ == "__main__":
    main()
