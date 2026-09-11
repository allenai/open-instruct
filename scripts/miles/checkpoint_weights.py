"""Exact, bounded-memory checks for a stream of converted checkpoint tensors."""

import hashlib
import json
from collections.abc import Mapping
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open


class SafeTensorState(Mapping):
    """Map checkpoint names to lazily read tensors, rejecting duplicate names."""

    def __init__(self, directory):
        self.stack = ExitStack()
        self.files = {}
        try:
            paths = sorted(Path(directory).glob("*.safetensors"))
            if not paths:
                raise FileNotFoundError(f"No safetensors in {directory}")
            for path in paths:
                handle = self.stack.enter_context(safe_open(path, framework="pt", device="cpu"))
                for name in handle.keys():  # noqa: SIM118 - safetensors handle is not iterable
                    if name in self.files:
                        raise ValueError(f"Duplicate checkpoint tensor: {name}")
                    self.files[name] = handle
        except BaseException:
            self.stack.close()
            raise

    def __getitem__(self, name):
        return self.files[name].get_tensor(name)

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stack.close()


def compare_stream(stream, reference, *, native_vocab_size=None, hf_vocab_size=None):
    """Compare every value after the explicitly recorded export dtype conversion.

    Only embeddings and the LM head may lose padding rows, and only when both
    vocabulary sizes are supplied. No arbitrary reshape or tolerance is allowed.
    """
    seen = set()
    digest = hashlib.sha256()
    total_bytes = 0
    casts = set()
    trimmed = []
    special = {}
    for name, actual in stream:
        if name in seen:
            raise ValueError(f"Duplicate exported tensor: {name}")
        if name not in reference:
            raise ValueError(f"Unexpected exported tensor: {name}")
        seen.add(name)
        expected = reference[name]
        if (
            name in ("model.embed_tokens.weight", "lm_head.weight")
            and native_vocab_size is not None
            and hf_vocab_size is not None
            and native_vocab_size > hf_vocab_size
        ):
            if actual.shape[0] != native_vocab_size or expected.shape[0] != hf_vocab_size:
                raise ValueError(f"Vocabulary size does not match checkpoint tensors: {name}")
            actual = actual[:hf_vocab_size]
            trimmed.append(name)
        if actual.shape != expected.shape:
            raise ValueError(f"Shape mismatch for {name}: {tuple(actual.shape)} != {tuple(expected.shape)}")
        if actual.dtype != expected.dtype:
            casts.add(f"{actual.dtype}->{expected.dtype}")
        actual = actual.detach().to(device="cpu", dtype=expected.dtype).contiguous()
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        if not torch.equal(actual, expected):
            error = (actual.float() - expected.float()).abs().max().item()
            raise ValueError(f"Value mismatch for {name}: max_abs_error={error}")
        digest.update(json.dumps([name, list(actual.shape), str(actual.dtype)]).encode())
        digest.update(actual.view(torch.uint8).numpy().tobytes())
        total_bytes += actual.numel() * actual.element_size()
        if name.endswith((".q_norm.weight", ".k_norm.weight", ".ssmax_scale")):
            special[name] = list(actual.shape)
    missing = set(reference) - seen
    if missing:
        raise ValueError(f"Missing exported tensors: {sorted(missing)}")
    if not seen:
        raise ValueError("Empty checkpoint comparison")
    return {
        "exact_match_after_export_cast": True,
        "tensor_count": len(seen),
        "bytes": total_bytes,
        "ordered_tensor_content_sha256": digest.hexdigest(),
        "dtype_conversions": sorted(casts),
        "trimmed_vocabulary_tensors": trimmed,
        "attention_parameter_shapes": special,
    }
