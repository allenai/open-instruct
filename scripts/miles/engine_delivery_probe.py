"""Two-GPU frozen-bucket/standalone-communicator probe; no model or WEKA needed."""

import json
import uuid
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import ray
import torch
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from torch import distributed as dist

from open_instruct.miles import actor, engine_delivery
from open_instruct.miles.state import PolicyClock


def fixture(device):
    return {
        "w": torch.arange(4_194_304, device=device, dtype=torch.float32).reshape(1024, 4096).T,
        "bias": torch.arange(2048, device=device, dtype=torch.bfloat16),
    }


@ray.remote(num_gpus=1)
class Source:
    def __init__(self):
        torch.cuda.set_device(0)
        dist.init_process_group(
            "gloo", init_method=f"file:///tmp/drain-probe-{uuid.uuid4().hex}", rank=0, world_size=1
        )
        self.clock = PolicyClock()
        self.weights = fixture("cuda")

    def location(self):
        return actor.OLMoCoreTrainRayActor.delivery_location(self)

    def snapshot(self):
        self.args = SimpleNamespace(
            update_weight_buffer_size=1024**2,
            olmo_core=SimpleNamespace(stream_moe_export=True, expert_publication="fused"),
        )
        self.train_module = self.hf_config = None
        self._agree = lambda operation: operation()
        with mock.patch.object(
            engine_delivery.models, "iter_export_state", lambda *a, **k: iter(self.weights.items())
        ):
            return engine_delivery.capture(self)

    def advance(self):
        for tensor in self.weights.values():
            tensor.add_(10000)
        self.clock.completed_steps += 1
        return self.clock.completed_steps


@ray.remote(num_gpus=1)
class Receiver:
    """Exercise the real wire layout; deliberately no SGLang loader claim."""

    def __init__(self):
        torch.cuda.set_device(0)
        self.group = None
        self.weights = {}
        self.version = -1
        self.ended = self.flushed = False

    def init_weights_update_group(self, address, port, rank_offset, world_size, group_name, backend):
        self.group = engine_delivery.distributed_utils.init_process_group(
            backend=backend,
            init_method=f"tcp://{address}:{port}",
            rank=rank_offset,
            world_size=world_size,
            group_name=group_name,
            timeout=timedelta(seconds=30),
        )
        return {"success": True}

    def begin_weight_update(self):
        self.ended = self.flushed = False
        return {"success": True}

    def _make_request(self, endpoint, payload):
        assert endpoint == "update_weights_from_distributed"
        sizes = [
            torch.Size(shape).numel() * torch.empty((), dtype=getattr(torch, dtype)).element_size()
            for shape, dtype in zip(payload["shapes"], payload["dtypes"])
        ]
        flat = torch.empty(sum(sizes), device="cuda", dtype=torch.uint8)
        dist.broadcast(flat, 0, group=self.group)
        offset = 0
        for name, shape, dtype, size in zip(payload["names"], payload["shapes"], payload["dtypes"], sizes):
            self.weights[name] = flat[offset : offset + size].view(getattr(torch, dtype)).reshape(shape).clone()
            offset += size
        self.version = int(payload["weight_version"])
        return {"success": True}

    def end_weight_update(self):
        self.ended = True
        return {"success": True}

    def flush_cache(self):
        self.flushed = True
        return {"success": True}

    def get_weight_version(self):
        assert self.ended and self.flushed
        return str(self.version)

    def compare(self):
        expected = fixture("cpu")
        return set(self.weights) == set(expected) and all(
            torch.equal(self.weights[name].cpu(), value.bfloat16()) for name, value in expected.items()
        )

    def destroy_weights_update_group(self, name):
        dist.destroy_process_group(self.group)
        return {"success": True}


def main():
    ray.init(num_gpus=2, num_cpus=8, include_dashboard=False, object_store_memory=1024**3)
    source, receiver = Source.remote(), Receiver.remote()
    sender = None
    try:
        location = ray.get(source.location.remote(), timeout=60)
        sender = engine_delivery.EngineDelivery.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(location["node_id"], soft=False),
            runtime_env={
                "env_vars": {
                    "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    "CUDA_VISIBLE_DEVICES": location["cuda_visible_devices"],
                }
            },
        ).remote(receiver, location["cuda_visible_devices"], 0, 30)
        ray.get(sender.connect.remote(), timeout=90)
        snapshot = ray.get(source.snapshot.remote(), timeout=30)
        assert len(snapshot.buckets) == 2
        assert ray.get(source.advance.remote(), timeout=30) == 1
        result = ray.get(sender.deliver.remote(snapshot), timeout=60)
        assert result["version"] == 0
        assert ray.get(receiver.compare.remote(), timeout=30)
        ray.get(sender.close.remote(), timeout=60)
        report = {
            "passed": True,
            "source": location,
            "delivery": result,
            "scope": "real Ray object store, separate CUDA processes and NCCL; synthetic receiver, no model",
        }
        Path("/output/probe.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        if sender is not None:
            ray.kill(sender)
        ray.kill(source)
        ray.kill(receiver)
        ray.shutdown()


if __name__ == "__main__":
    main()
