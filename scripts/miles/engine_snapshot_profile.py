"""Isolate 37 GB snapshot copying from native model export and serving startup."""

import json
import time
import uuid
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest import mock

import ray
import torch
from torch import distributed as dist

from open_instruct.miles import engine_delivery
from open_instruct.miles.state import PolicyClock


@ray.remote(num_gpus=1)
class Source:
    def __init__(self):
        torch.cuda.set_device(0)
        dist.init_process_group(
            "gloo", init_method=f"file:///tmp/snapshot-profile-{uuid.uuid4().hex}", rank=0, world_size=1
        )
        # Representative fused expert tensor sizes, then the remaining model bytes.
        sizes = [1_247_805_440, 623_902_720] * 19
        sizes += [37_028_386_304 - sum(sizes)]
        self.weights = {
            f"tensor.{i}": torch.ones(size // 2, dtype=torch.bfloat16, device="cuda") for i, size in enumerate(sizes)
        }
        self.args = SimpleNamespace(
            update_weight_buffer_size=1024**3,
            olmo_core=SimpleNamespace(stream_moe_export=True, expert_publication="fused"),
        )
        self.clock = PolicyClock()
        self.train_module = self.hf_config = None
        self._agree = lambda fn: fn()

    def measure(self):
        timings = defaultdict(float)
        calls = defaultdict(int)
        bucket_class = engine_delivery.update_weight_utils.FlattenedTensorBucket
        original_pack, original_cpu, original_put = bucket_class.__init__, torch.Tensor.cpu, ray.put

        def pack(bucket, *args, **kwargs):
            started = time.perf_counter()
            torch.cuda.synchronize()
            timings["prior_cuda_wait"] += time.perf_counter() - started
            started = time.perf_counter()
            value = original_pack(bucket, *args, **kwargs)
            torch.cuda.synchronize()
            timings["gpu_pack"] += time.perf_counter() - started
            calls["gpu_pack"] += 1
            return value

        def cpu(tensor, *args, **kwargs):
            started = time.perf_counter()
            value = original_cpu(tensor, *args, **kwargs)
            timings["d2h"] += time.perf_counter() - started
            calls["d2h"] += 1
            return value

        def put(value, *args, **kwargs):
            started = time.perf_counter()
            result = original_put(value, *args, **kwargs)
            timings["ray_put"] += time.perf_counter() - started
            calls["ray_put"] += 1
            return result

        torch.cuda.synchronize()
        with (
            mock.patch.object(engine_delivery.models, "iter_export_state", lambda *a, **k: iter(self.weights.items())),
            mock.patch.object(bucket_class, "__init__", pack),
            mock.patch.object(torch.Tensor, "cpu", cpu),
            mock.patch.object(ray, "put", put),
        ):
            snapshot = engine_delivery.capture(self)
        assert snapshot.nbytes == 37_028_386_304
        frozen = ray.get(snapshot.buckets[0][1])
        assert not frozen.flags.writeable
        result = {
            "total_seconds": snapshot.capture_seconds,
            "bytes": snapshot.nbytes,
            "phase_seconds": dict(timings),
            "calls": dict(calls),
            "unattributed_seconds": snapshot.capture_seconds - sum(timings.values()),
        }
        self.clock.completed_steps += 1
        return result


def main():
    ray.init(num_cpus=8, num_gpus=1, include_dashboard=False, object_store_memory=80 * 1024**3)
    worker = cast(Any, Source).remote()
    try:
        rows = [ray.get(worker.measure.remote(), timeout=600) for _ in range(3)]
        result = {
            "scope": "Synthetic BF16 source; exact capture implementation, excludes native export/EP collectives, serving and concurrent training. GPU synchronization added to isolate phases.",
            "measurements": rows,
        }
        Path("/output/snapshot-profile.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        ray.kill(worker)
        ray.shutdown()


if __name__ == "__main__":
    main()
