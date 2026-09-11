"""CPU drift of routers, normalization and KDA scalars from completed native saves."""

import argparse
import hashlib
import inspect
import json
import resource
import time
from collections.abc import Mapping
from pathlib import Path

import torch
from olmo_core.nn.hf import convert
from olmo_core.nn.moe.v2.hf.configuration_olmo3moe import Olmo3MoeConfig
from scripts.miles import checkpoint_drift, core_checkpoint_stream, megatron_checkpoint_stream
from scripts.miles.checkpoint_weights import SafeTensorState


def selected(name):
    return "router" in name or "norm" in name or name.endswith(("A_log", "dt_bias"))


class SmallNativeState(Mapping):
    """Use meta placeholders for excluded tensors while retaining converter checks."""

    def __init__(self, native):
        self.native = native
        self.bytes_read = 0
        self.keys_read = []

    def __iter__(self):
        return iter(self.native)

    def __len__(self):
        return len(self.native)

    def __getitem__(self, key):
        name, start, size = self.native.virtual[key]
        shape, dtype = self.native.parameters[name]
        if selected(key):
            if name not in self.keys_read:
                self.keys_read.append(name)
                self.bytes_read += torch.Size(shape).numel() * 4
            return self.native[key]
        value = torch.empty(shape, device="meta", dtype=dtype)
        return value if start is None else value.narrow(0, start, size)


def read_core_subset(native):
    state = SmallNativeState(native)
    result, inventory = {}, set()
    for name, tensor in convert.iter_olmo3moe_state_to_hf(native.hf_config, state):
        if name in inventory:
            raise ValueError(f"Duplicate converted name: {name}")
        inventory.add(name)
        if selected(name):
            if tensor.device.type != "cpu":
                raise ValueError(f"Selected output was not read: {name}")
            result[name] = tensor
        elif tensor.device.type != "meta":
            raise ValueError(f"Read an excluded tensor: {name}")
    return result, inventory, {"master_payload_bytes": state.bytes_read, "native_keys": state.keys_read}


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_boundary(core, megatron, updates):
    core_manifest = json.loads((core / "complete.json").read_text())
    mega_manifest = json.loads((megatron / "diagnostic-retention.json").read_text())
    if core_manifest["clock"]["completed_steps"] != updates or mega_manifest["completed_updates"] != updates:
        raise ValueError("Checkpoint completed-update boundaries differ")
    if mega_manifest["iteration"] != updates - 1:
        raise ValueError("Megatron iteration is not the requested completed-update boundary")
    cursor = core.parent.parent / "rollout" / f"global_dataset_state_dict_{updates - 1}.pt"
    if hash_file(cursor) != core_manifest["cursor_sha256"]:
        raise ValueError("Core committed cursor changed")
    for entry in mega_manifest["files"]:
        path = megatron / entry["path"]
        if not path.is_relative_to(megatron) or ".." in Path(entry["path"]).parts or path.is_symlink():
            raise ValueError("Invalid retained checkpoint path")
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["bytes"], entry["mtime_ns"]):
            raise ValueError("Retained completed checkpoint changed")
    return core_manifest, mega_manifest


def run(reference_path, core_path, megatron_path, updates):
    started = time.monotonic()
    core_manifest, mega_manifest = validate_boundary(core_path, megatron_path, updates)
    config = Olmo3MoeConfig(**core_manifest["hf_config"])
    native = core_checkpoint_stream.CoreCheckpointState(
        core_path / "model", config, category="reconstructed_model_storage"
    )
    core, inventory, reads = read_core_subset(native)
    megatron = megatron_checkpoint_stream.MegatronCheckpointState(megatron_path, config, max_cache_bytes=64 * 1024**2)
    with SafeTensorState(reference_path) as initial:
        if inventory != set(initial) or inventory != set(megatron):
            raise ValueError("Full canonical inventories differ")
        expected = {name for name in initial if selected(name)}
        if set(core) != expected:
            raise ValueError("Selected canonical inventory differs")
        reference = {name: initial[name] for name in sorted(expected)}
        mega = {name: megatron[name] for name in sorted(expected)}
        result = checkpoint_drift.compare(reference, core, mega)
    result.update(
        completed_updates=updates,
        scope="Router, normalization and KDA scalar model-visible drift only; Core reconstructed per native storage dtype; no Megatron optimizer masters or expert/attention matrices.",
        input_paths={"reference": str(reference_path), "core": str(core_path), "megatron": str(megatron_path)},
        core_reads=reads,
        megatron_reads={"payload_bytes": megatron.bytes_read, "tensor_reads": megatron.tensor_reads},
        seconds=time.monotonic() - started,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        metadata_sha256={
            "core": hash_file(core_path / "model/.metadata"),
            "megatron": hash_file(megatron_path / ".metadata"),
        },
        completion_sha256={
            "core": hash_file(core_path / "complete.json"),
            "megatron": hash_file(megatron_path / "diagnostic-retention.json"),
        },
        config_sha256=hash_file(reference_path / "config.json"),
        conversion_source_sha256={
            name: hash_file(inspect.getfile(module))
            for name, module in {
                "core_converter": convert,
                "core_reader": core_checkpoint_stream,
                "megatron_reader": megatron_checkpoint_stream,
                "megatron_exporter": megatron_checkpoint_stream.OlmoDirectWeightExporter,
            }.items()
        },
    )
    validate_boundary(core_path, megatron_path, updates)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "core", "megatron", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    deadline = time.monotonic() + args.wait_seconds
    markers = (args.core / "complete.json", args.megatron / "diagnostic-retention.json")
    while not all(path.is_file() for path in markers):
        if time.monotonic() >= deadline:
            raise TimeoutError("Completed native snapshots did not arrive before the deadline")
        print(json.dumps({"waiting_for": [str(path) for path in markers if not path.is_file()]}), flush=True)
        time.sleep(min(30, max(0, deadline - time.monotonic())))
    result = run(args.reference, args.core, args.megatron, args.updates)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("completed_updates", "scope", "seconds", "peak_rss_kib", "core_reads", "megatron_reads")
            }
        )
    )


if __name__ == "__main__":
    main()
