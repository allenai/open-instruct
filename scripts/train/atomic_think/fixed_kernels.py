"""Replay verified forward-kernel choices for H077 inference only."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from triton.runtime import autotuner

REFERENCE = Path("scripts/train/atomic_think/kernel_reference.json")


def tensor_tree(value):
    if isinstance(value, torch.Tensor):
        raw = value.detach().contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(value, dict):
        return {key: tensor_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [tensor_tree(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported capture value: {type(value)}")


def capture_signatures(capture):
    documents = capture["documents"]
    signatures = {
        doc["id"]: hashlib.sha256(json.dumps(tensor_tree(doc), sort_keys=True, allow_nan=False).encode()).hexdigest()
        for doc in documents
    }
    if len(signatures) != len(documents):
        raise ValueError("Duplicate capture document")
    return signatures


class FixedKernels:
    """Reject uncalibrated keys, preserve original Config objects and kernel hooks."""

    def __init__(self, path=REFERENCE):
        path = Path(path)
        self.reference_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        self.reference = json.loads(path.read_text())
        if self.reference["version"] != 1:
            raise ValueError("Unknown kernel-reference schema")
        entries = self.reference["tuning_choices"]
        self.choices = {f"{entry['function']}:{entry['key']}": entry for entry in entries}
        if len(self.choices) != len(entries):
            raise ValueError("Duplicate reference tuning key")
        self.observed = set()
        self.original_run = None

    def validate_runtime(self):
        expected = self.reference["runtime"]
        packages = {name: importlib.metadata.version(name) for name in expected["packages"]}
        gpu = torch.cuda.get_device_properties(0)
        actual = {
            "packages": packages,
            "cuda": torch.version.cuda,
            "gpu_name": gpu.name,
            "capability": [gpu.major, gpu.minor],
            "multiprocessors": gpu.multi_processor_count,
            "matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        }
        if actual != expected:
            raise ValueError(f"Inference runtime differs from kernel calibration: {actual}")

    def select(self, tuner, args, kwargs):
        # Match the key construction in the pinned Triton 3.6.0 Autotuner.run.
        all_args = {**dict(zip(tuner.arg_names, args)), **kwargs}
        named = {key: value for key, value in all_args.items() if key in tuner.arg_names}
        key = tuple(
            [named[name] for name in tuner.keys if name in named]
            + [str(value.dtype) for value in named.values() if hasattr(value, "dtype")]
        )
        function = f"{tuner.base_fn.__module__}.{tuner.base_fn.__qualname__}"
        identity = f"{function}:{key!r}"
        if identity not in self.choices:
            raise ValueError(f"Uncalibrated inference kernel key: {identity}")
        expected = self.choices[identity]
        matches = [
            config
            for config in tuner.configs
            if config.all_kwargs() == expected["kwargs"]
            and (
                None
                if config.pre_hook is None
                else getattr(config.pre_hook, "__qualname__", type(config.pre_hook).__name__)
            )
            == expected["pre_hook"]
        ]
        if not matches:
            raise ValueError(f"Recorded kernel config no longer available: {identity}")
        config = matches[0]
        if len(tuner.configs) == 1 and tuner.configs[0].all_kwargs() != config.all_kwargs():
            raise ValueError("Single-config kernel differs from calibration")
        tuner.cache[key] = config
        self.observed.add(identity)
        return config

    def __enter__(self):
        self.validate_runtime()
        if self.original_run is not None:
            raise RuntimeError("Kernel replay context is already active")
        self.original_run = autotuner.Autotuner.run

        def run(tuner, *args, **kwargs):
            selected = self.select(tuner, args, kwargs)
            result = self.original_run(tuner, *args, **kwargs)
            if tuner.best_config.all_kwargs() != selected.all_kwargs():
                raise RuntimeError("Kernel execution did not use the pinned choice")
            return result

        autotuner.Autotuner.run = run
        return self

    def __exit__(self, *exception):
        autotuner.Autotuner.run = self.original_run
        self.original_run = None

    def receipt(self):
        return {
            "reference_sha256": self.reference_hash,
            "observed_keys": sorted(self.observed),
            "runtime": self.reference["runtime"],
        }
