"""Snapshot capture and per-engine delivery, isolated from trainer communicators.

Host buckets live once in Ray's immutable object store. Delivery actors share the
source trainer's GPU only for one bucket at a time; they never read its parameters
or join its process groups. Each actor owns a two-rank NCCL group with one TP1 engine.
"""

import os
import socket
import time
import uuid
from datetime import timedelta

import numpy as np
import ray
import torch
from miles.backends.fsdp_utils import update_weight_utils
from miles.utils import async_utils, distributed_utils
from torch import distributed as dist

from open_instruct.miles.publication.engine_drain import WeightSnapshot
from open_instruct.miles.training import models


def copy_bucket_to_host(actor, flat):
    """Reuse pinned staging; Ray must finish freezing it before the next copy."""
    if not flat.is_cuda:
        return flat.cpu()
    staging = getattr(actor, "_engine_drain_host_staging", None)
    if staging is None or staging.numel() < flat.numel():
        staging = torch.empty(flat.numel(), dtype=flat.dtype, device="cpu", pin_memory=True)
        actor._engine_drain_host_staging = staging
    packed = staging[: flat.numel()]
    packed.copy_(flat, non_blocking=True)
    # Freeze only after the D2H completes. The exported GPU bucket can then be
    # released, and ray.put copies the staging bytes into immutable object storage.
    torch.cuda.current_stream(flat.device).synchronize()
    return packed


def capture(actor):
    """Collective optimizer-boundary export; only rank zero retains frozen bytes."""
    started = time.perf_counter()
    refs, bucket, size, total = [], [], 0, 0
    names_seen = set()

    def flush():
        if dist.get_rank() != 0:
            return
        # Pack on the source device and perform one D2H copy per bucket. Doing
        # CPU concatenation and a blocking copy for every individual parameter
        # cost 44 seconds for the first measured 37 GB snapshot.
        flat = update_weight_utils.FlattenedTensorBucket(named_tensors=bucket).get_flattened_tensor()
        packed = copy_bucket_to_host(actor, flat)
        # ray.put serializes before returning. The resulting NumPy object is
        # read-only on retrieval; subsequent optimizer writes cannot alias it.
        refs.append(
            (
                tuple((name, tuple(t.shape), str(t.dtype).removeprefix("torch.")) for name, t in bucket),
                ray.put(packed.numpy()),
            )
        )

    for name, tensor in models.iter_export_state(
        actor.train_module,
        actor.hf_config,
        stream_moe=actor.args.olmo_core.stream_moe_export,
        fused_experts=actor.args.olmo_core.expert_publication == "fused",
    ):
        if name in names_seen or tensor.numel() == 0:
            raise ValueError(f"invalid snapshot tensor {name}")
        names_seen.add(name)
        nbytes = tensor.numel() * 2
        if size and size + nbytes > actor.args.update_weight_buffer_size:
            actor._agree(flush)
            bucket, size = [], 0
        if dist.get_rank() == 0:
            # Exporters may reuse scratch storage on the next iteration. Own
            # each tensor immediately, but retain at most one GPU bucket rather
            # than a complete extra model. The packed CPU array is frozen by Ray.
            bucket.append((name, tensor.detach().to(dtype=torch.bfloat16, copy=True).contiguous()))
        size += nbytes
        total += nbytes
    if size:
        actor._agree(flush)
    dist.barrier()
    actor.clock.snapshot_ready_step = actor.clock.completed_steps
    if dist.get_rank() == 0:
        return WeightSnapshot(actor.clock.completed_steps, tuple(refs), total, time.perf_counter() - started)
    return None


@ray.remote(num_cpus=0, num_gpus=0, max_restarts=0, max_task_retries=0)
class EngineDelivery:
    """One process and communicator per engine; bounded RPCs, no fleet membership."""

    def __init__(self, engine, cuda_visible_devices, device_index, timeout):
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        torch.cuda.set_device(device_index)
        self.engine = engine
        self.timeout = timeout
        self.group_name = f"core-drain-{uuid.uuid4().hex}"
        self.group = None

    def connect(self):
        address = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        pending = async_utils.submit(
            self.engine.init_weights_update_group(address, port, 1, 2, self.group_name, backend="nccl")
        )
        self.group = distributed_utils.init_process_group(
            backend="nccl",
            init_method=f"tcp://{address}:{port}",
            world_size=2,
            rank=0,
            group_name=self.group_name,
            timeout=timedelta(seconds=self.timeout),
        )
        pending.result(timeout=self.timeout)

    def _call(self, method, *args, **kwargs):
        result = async_utils.submit(getattr(self.engine, method)(*args, **kwargs)).result(timeout=self.timeout)
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError(f"engine rejected {method}: {result}")
        return result

    def deliver(self, snapshot):
        started = time.perf_counter()
        # Admission is already closed and all reserved requests have returned.
        # No pause/abort call is needed. A failed handshake never reopens admission.
        self._call("begin_weight_update")
        transferred = 0
        for metadata, reference in snapshot.buckets:
            array = ray.get(reference, timeout=self.timeout)
            # Never modify the read-only object-store array. Only the GPU copy
            # is passed to NCCL. Blocking H2D ensures its lifetime is sufficient.
            host = torch.empty(array.shape, dtype=torch.uint8, pin_memory=True)
            np.copyto(host.numpy(), array)
            flat = host.to(device="cuda", non_blocking=False)
            del host
            payload = dict(
                names=[item[0] for item in metadata],
                shapes=[item[1] for item in metadata],
                dtypes=[item[2] for item in metadata],
                group_name=self.group_name,
                weight_version=str(snapshot.version),
                load_format="flattened_bucket",
                flush_cache=False,
            )
            pending = async_utils.submit(self.engine._make_request("update_weights_from_distributed", payload))
            dist.broadcast(flat, 0, group=self.group, async_op=True).wait()
            result = pending.result(timeout=self.timeout)
            if not isinstance(result, dict) or result.get("success") is not True:
                raise RuntimeError(f"engine rejected snapshot bucket: {result}")
            transferred += flat.numel()
            del flat, array
        self._call("end_weight_update")
        self._call("update_weight_version", str(snapshot.version), abort_all_requests=False)
        self._call("flush_cache")
        version = int(self._call("get_weight_version"))
        if version != snapshot.version or transferred != snapshot.nbytes:
            raise RuntimeError("incomplete snapshot publication or incorrect engine version")
        return {
            "version": version,
            "bytes": transferred,
            "delivery_seconds": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        }

    def close(self):
        if self.group is not None:
            pending = async_utils.submit(self.engine.destroy_weights_update_group(self.group_name))
            dist.destroy_process_group(self.group)
            self.group = None
            pending.result(timeout=self.timeout)
