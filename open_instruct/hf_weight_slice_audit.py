"""Bounded read-only original/export tensor observations for historical OPD.

This samples CPU slices; it never loads a full model or restores an optimizer.
Equal slices do not establish full weight equality or serving-transfer parity.
"""

import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


def checked_file(root, name, *, cache_links=False):
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("Weight inventory must use a plain filename")
    path = root / name
    if path.is_symlink():
        cache_root = root.parent.parent
        if (
            not cache_links
            or root.parent.name != "snapshots"
            or not path.resolve().is_relative_to(cache_root / "blobs")
        ):
            raise ValueError("Unexpected weight/config symlink")
    if not path.is_file():
        raise ValueError("Missing weight/config file")
    return path


def inventory(root, *, cache_links=False):
    root = Path(root).resolve()
    config = checked_file(root, "config.json", cache_links=cache_links)
    if config.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Config exceeds bound")
    index = root / "model.safetensors.index.json"
    mapping = {}
    paths = [config]
    if index.exists():
        checked_file(root, index.name, cache_links=cache_links)
        if index.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("Weight index exceeds bound")
        mapping = json.loads(index.read_text())["weight_map"]
        paths.append(index)
    else:
        shards = sorted(root.glob("*.safetensors"))
        if not 1 <= len(shards) <= 64:
            raise ValueError("Weight shard count exceeds bound")
        for path in shards:
            checked_file(root, path.name, cache_links=cache_links)
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                keys = handle.keys()
                for key in keys:
                    if key in mapping:
                        raise ValueError("Duplicate weight key")
                    mapping[key] = path.name
    if not isinstance(mapping, dict) or not 1 <= len(mapping) <= 10000:
        raise ValueError("Invalid bounded weight map")
    if any(not isinstance(key, str) or not isinstance(name, str) for key, name in mapping.items()):
        raise ValueError("Invalid weight map key/filename")
    names = sorted(set(mapping.values()))
    if not 1 <= len(names) <= 64:
        raise ValueError("Weight shard count exceeds bound")
    paths.extend(checked_file(root, name, cache_links=cache_links) for name in names)
    stats = {
        path.name: {
            "resolved_path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in paths
    }
    return mapping, stats, hashlib.sha256(config.read_bytes()).hexdigest()


def selected_keys(mapping):
    """Six distinct deterministic families present in the Qwen3.5 text model."""

    def matches(suffix):
        return sorted(
            (key for key in mapping if key.endswith(suffix)),
            key=lambda key: [int(piece) if piece.isdigit() else piece for piece in re.split(r"(\d+)", key)],
        )

    embeddings = matches(".embed_tokens.weight")
    if len(embeddings) != 1:
        raise ValueError("Expected one text embedding tensor")
    text_prefix = embeddings[0].removesuffix(".embed_tokens.weight")
    norm = f"{text_prefix}.norm.weight"
    queries = [key for key in matches(".self_attn.q_proj.weight") if key.startswith(f"{text_prefix}.layers.")]
    downs = [key for key in matches(".mlp.down_proj.weight") if key.startswith(f"{text_prefix}.layers.")]
    if norm not in mapping or len(queries) < 2 or len(downs) < 2:
        raise ValueError("Expected embedding/norm and distinct first/last attention/MLP tensors")
    return [embeddings[0], norm, queries[0], queries[-1], downs[0], downs[-1]]


def read_slice(root, mapping, key, *, cache_links=False):
    path = checked_file(root, mapping[key], cache_links=cache_links)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        view = handle.get_slice(key)
        shape = view.get_shape()
        if len(shape) not in (1, 2) or any(size <= 0 for size in shape):
            raise ValueError("Unsupported tensor shape")
        starts = sorted({0, max(0, shape[0] // 2 - 4), max(0, shape[0] - 8)})
        slices = [view[start : start + 8, :16] if len(shape) == 2 else view[start : start + 8] for start in starts]
        values = torch.cat([value.reshape(-1).float() for value in slices])
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Nonfinite sampled weights")
    return shape, starts, values


def compare(original, export, *, cache_links=False):
    original, export = Path(original).resolve(), Path(export).resolve()
    if original == export or original.is_relative_to(export) or export.is_relative_to(original):
        raise ValueError("Original and final roots must be distinct")
    before, before_stats, before_config = inventory(original, cache_links=cache_links)
    after, after_stats, after_config = inventory(export)
    keys = selected_keys(before)
    records = []
    for key in keys:
        if key not in after:
            raise ValueError("Final export lacks selected original key")
        shape, starts, initial = read_slice(original, before, key, cache_links=cache_links)
        final_shape, final_starts, final = read_slice(export, after, key)
        if shape != final_shape or starts != final_starts:
            raise ValueError("Original/final shape mismatch")
        delta = final - initial
        records.append(
            {
                "key": key,
                "shape": shape,
                "row_starts": starts,
                "max_rows": 8,
                "max_columns": 16,
                "sampled_elements": delta.numel(),
                "changed_sampled_elements": int(torch.count_nonzero(delta)),
                "max_abs_gap": float(delta.abs().max()),
                "original_slice_sha256": hashlib.sha256(initial.numpy().tobytes()).hexdigest(),
                "final_slice_sha256": hashlib.sha256(final.numpy().tobytes()).hexdigest(),
            }
        )
    if inventory(original, cache_links=cache_links)[1] != before_stats or inventory(export)[1] != after_stats:
        raise ValueError("Input file inventory changed during observation")
    return {
        "original_root": str(original),
        "export_root": str(export),
        "slices": records,
        "sampled_weight_movement": any(item["changed_sampled_elements"] > 0 for item in records),
        "original_config_sha256": before_config,
        "final_config_sha256": after_config,
        "original_inventory": before_stats,
        "final_inventory": after_stats,
        "source_stats_unchanged": True,
        "limits": "Six tensor families, three at-most 8x16 CPU windows each. Movement is final versus initial only; equal slices are not full equality. No optimizer/cursor restore or serving-transfer parity. Shard provenance uses size/mtime, not full checksums.",
    }
