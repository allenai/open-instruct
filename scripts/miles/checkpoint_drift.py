"""Bounded CPU drift measurements for explicitly canonical checkpoint tensors.

Native DCP inspection reads metadata only. Comparison requires already canonical
HF names and shapes; this module never guesses a native layout or casts masters.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from scripts.miles.checkpoint_weights import SafeTensorState
from torch.distributed.checkpoint import FileSystemReader


class DriftAccumulator:
    def __init__(self):
        self.count = 0
        self.reference_ss = self.core_ss = self.megatron_ss = self.difference_ss = self.dot = 0.0
        self.core_max = self.megatron_max = self.difference_max = 0.0
        self.core_changed = self.megatron_changed = 0
        self.dtypes = Counter()

    def add(self, reference, core, megatron, *, chunk_elements=1_048_576):
        if reference.shape != core.shape or reference.shape != megatron.shape:
            raise ValueError("Canonical shapes differ; implicit reshape is forbidden")
        if any(value.device.type != "cpu" for value in (reference, core, megatron)):
            raise ValueError("Drift analysis requires CPU tensors")
        if chunk_elements <= 0:
            raise ValueError("Chunk size must be positive")
        self.dtypes[(str(reference.dtype), str(core.dtype), str(megatron.dtype))] += reference.numel()
        flat = [value.reshape(-1) for value in (reference, core, megatron)]
        for start in range(0, reference.numel(), chunk_elements):
            initial, left, right = [value[start : start + chunk_elements].double() for value in flat]
            if not all(torch.isfinite(value).all().item() for value in (initial, left, right)):
                raise ValueError("Nonfinite checkpoint tensor")
            a, b = left - initial, right - initial
            delta = a - b
            self.count += initial.numel()
            self.reference_ss += initial.square().sum().item()
            self.core_ss += a.square().sum().item()
            self.megatron_ss += b.square().sum().item()
            self.difference_ss += delta.square().sum().item()
            self.dot += (a * b).sum().item()
            self.core_max = max(self.core_max, a.abs().max().item())
            self.megatron_max = max(self.megatron_max, b.abs().max().item())
            self.difference_max = max(self.difference_max, delta.abs().max().item())
            self.core_changed += torch.count_nonzero(a).item()
            self.megatron_changed += torch.count_nonzero(b).item()

    def merge(self, other):
        for key in (
            "count",
            "reference_ss",
            "core_ss",
            "megatron_ss",
            "difference_ss",
            "dot",
            "core_changed",
            "megatron_changed",
        ):
            setattr(self, key, getattr(self, key) + getattr(other, key))
        for key in ("core_max", "megatron_max", "difference_max"):
            setattr(self, key, max(getattr(self, key), getattr(other, key)))
        self.dtypes.update(other.dtypes)

    def report(self):
        if not self.count:
            raise ValueError("Empty parameter comparison")
        initial = math.sqrt(self.reference_ss)
        left, right, difference = map(math.sqrt, (self.core_ss, self.megatron_ss, self.difference_ss))
        return {
            "parameters": self.count,
            "reference_l2": initial,
            "core": {
                "change_l2": left,
                "change_rms": left / math.sqrt(self.count),
                "relative_to_reference_l2": left / initial if initial else None,
                "max_abs_change": self.core_max,
                "changed_fraction": self.core_changed / self.count,
            },
            "megatron": {
                "change_l2": right,
                "change_rms": right / math.sqrt(self.count),
                "relative_to_reference_l2": right / initial if initial else None,
                "max_abs_change": self.megatron_max,
                "changed_fraction": self.megatron_changed / self.count,
            },
            "between_backends": {
                "difference_l2": difference,
                "difference_rms": difference / math.sqrt(self.count),
                "relative_to_core_change_l2": difference / left if left else None,
                "relative_to_megatron_change_l2": difference / right if right else None,
                "change_direction_cosine": self.dot / (left * right) if left and right else None,
                "max_abs_difference": self.difference_max,
            },
            "dtype_parameters": [
                {"reference": names[0], "core": names[1], "megatron": names[2], "parameters": count}
                for names, count in sorted(self.dtypes.items())
            ],
        }


def parameter_group(name):
    if ".router." in name:
        return "router"
    if ".mlp.experts." in name:
        return "routed_experts"
    if ".shared_expert" in name:
        return "shared_experts"
    if "latent_" in name:
        return "latent_projections"
    if "norm" in name:
        return "normalization"
    if ".self_attn." in name:
        return "attention"
    if ".mlp." in name:
        return "dense_mlp"
    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        return "embedding_and_output"
    raise ValueError(f"Unclassified canonical parameter: {name}")


def compare(reference, core, megatron):
    if not set(reference) or set(reference) != set(core) or set(reference) != set(megatron):
        raise ValueError("Canonical inventories differ or are empty")
    totals = defaultdict(DriftAccumulator)
    tensors = {}
    for name in sorted(reference):
        measurement = DriftAccumulator()
        measurement.add(reference[name], core[name], megatron[name])
        tensors[name] = measurement.report()
        group = parameter_group(name)
        match = re.match(r"model\.layers\.(\d+)\.", name)
        layer = match.group(1) if match else "global"
        for key in ("all", f"group/{group}", f"layer/{layer}", f"layer/{layer}/group/{group}"):
            totals[key].merge(measurement)
    return {
        "schema_version": 1,
        "aggregates": {name: value.report() for name, value in sorted(totals.items())},
        "tensors": tensors,
        "interpretation": (
            "Changes are measured against the supplied initial HF values, not inferred optimizer-start masters. "
            "Original FP32 source values and explicit model-storage rounding must be recorded by native adapters. "
            "Float64 chunked accumulators; RMS normalizes parameter count; no acceptance tolerance is imposed. "
            "Zero denominators/directions are null. Core and Megatron denominators are both reported."
        ),
    }


def inspect_native(path):
    metadata_path = path / ".metadata"
    metadata = FileSystemReader(path).read_metadata()
    tensors = {}
    for name, entry in metadata.state_dict_metadata.items():
        if not hasattr(entry, "size"):
            continue
        tensors[name] = {
            "shape": list(entry.size),
            "dtype": str(entry.properties.dtype),
            "chunks": [{"offsets": list(chunk.offsets), "sizes": list(chunk.sizes)} for chunk in entry.chunks],
        }
    return {
        "path": str(path),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "tensor_metadata": tensors,
        "non_tensor_keys": sorted(set(metadata.state_dict_metadata) - set(tensors)),
        "scope": "Trusted native DCP metadata only; no tensor payload read or checkpoint mutation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inventory = commands.add_parser("inspect")
    inventory.add_argument("checkpoint", type=Path)
    comparison = commands.add_parser("compare-canonical")
    for name in ("reference", "core", "megatron"):
        comparison.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == "inspect":
        result = inspect_native(args.checkpoint)
    else:
        with (
            SafeTensorState(args.reference) as reference,
            SafeTensorState(args.core) as core,
            SafeTensorState(args.megatron) as megatron,
        ):
            result = compare(reference, core, megatron)
        result["input_paths"] = {name: str(getattr(args, name)) for name in ("reference", "core", "megatron")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
