"""Cache exact packed samples without changing dataset identity or loader order."""

import concurrent.futures
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from olmo_core.data import NumpyPackedFSLDataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)
FIELDS = {"input_ids", "label_mask", "doc_lens"}
DOC_DTYPES = {"torch.int32": torch.int32, "torch.int64": torch.int64}


def stage_cache(source, destination):
    """Copy a completed cache to node-local storage and verify every payload byte."""
    source, destination = Path(source), Path(destination)
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["version"] != 1 or set(manifest["files"]) != {"input_ids.npy", "label_mask.npy"}:
        raise ValueError("Unsupported packed cache files")
    destination.mkdir(parents=True, exist_ok=False)
    for name, expected in manifest["files"].items():
        path = destination / name
        shutil.copyfile(source / name, path)
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Packed cache staging hash mismatch: {name}")
    (destination / "manifest.json").write_bytes(manifest_bytes)
    return {
        "source": str(source),
        "destination": str(destination),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "files": manifest["files"],
    }


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def bounded_samples(dataset, indices, workers):
    remaining = iter(enumerate(indices))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        for _ in range(2 * workers):
            if (entry := next(remaining, None)) is not None:
                row, index = entry
                pending[pool.submit(dataset.__getitem__, index)] = row
        while pending:
            done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                row = pending.pop(future)
                yield row, future.result()
                if (entry := next(remaining, None)) is not None:
                    new_row, index = entry
                    pending[pool.submit(dataset.__getitem__, index)] = new_row


def build_cache(dataset, indices, output, *, provenance, workers=8, resume=False, checkpoint_interval=128):
    """Materialize a bounded selection; publish the manifest only after verification."""
    output = Path(output)
    indices = [int(index) for index in indices]
    if not indices or len(set(indices)) != len(indices) or workers < 1 or checkpoint_interval < 1:
        raise ValueError("Require unique nonempty indices and positive worker count")
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Sample index outside source dataset")
    # Materialize lazy metadata once before sharing this read-only dataset with threads.
    _ = dataset.source_size_groups, dataset.source_instance_offsets
    shape = (len(indices), dataset.sequence_length)
    identity = {
        "version": 1,
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_instances": len(dataset),
        "sequence_length": dataset.sequence_length,
        "source_paths": [str(path) for path in dataset.paths],
        "label_mask_paths": [str(path) for path in (dataset._label_mask_paths or [])],
        "provenance": provenance,
    }
    progress_path = output / "progress.json"
    if resume:
        if (output / "manifest.json").exists():
            raise FileExistsError("Cache is already complete; consume its manifest instead of rebuilding")
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity or progress["indices"] != indices:
            raise ValueError("Incomplete cache belongs to another dataset or sample selection")
        tokens = np.load(output / "input_ids.npy", mmap_mode="r+", allow_pickle=False)
        masks = np.load(output / "label_mask.npy", mmap_mode="r+", allow_pickle=False)
        if tokens.shape != shape or masks.shape != shape or tokens.dtype != np.uint32 or masks.dtype != np.bool_:
            raise ValueError("Invalid incomplete cache arrays")
        rows = progress["rows"]
        if len(rows) != len(indices):
            raise ValueError("Invalid incomplete cache row count")
        for row, entry in enumerate(rows):
            if entry is None:
                continue
            restored = {
                "input_ids": torch.tensor(tokens[row].astype(np.int64)),
                "label_mask": torch.tensor(masks[row]),
                "doc_lens": torch.tensor(entry["doc_lens"], dtype=DOC_DTYPES[entry["doc_lens_dtype"]]),
            }
            if entry["index"] != indices[row] or entry["hashes"] != {
                key: tensor_hash(value) for key, value in restored.items()
            }:
                raise ValueError("Incomplete cache row failed integrity verification")
    else:
        output.mkdir(parents=True, exist_ok=False)
        tokens = np.lib.format.open_memmap(output / "input_ids.npy", mode="w+", dtype=np.uint32, shape=shape)
        masks = np.lib.format.open_memmap(output / "label_mask.npy", mode="w+", dtype=np.bool_, shape=shape)
        rows = [None] * len(indices)

    def save_progress():
        tokens.flush()
        masks.flush()
        temporary = output / "progress.partial.json"
        temporary.write_text(
            json.dumps({"identity": identity, "indices": indices, "rows": rows}, separators=(",", ":")) + "\n"
        )
        temporary.replace(progress_path)

    save_progress()
    pending_rows = [row for row, entry in enumerate(rows) if entry is None]
    completed = len(rows) - len(pending_rows)
    logger.info("Verified %s existing cached samples; %s remain", completed, len(pending_rows))
    pending_indices = [indices[row] for row in pending_rows]
    for finished, (pending_row, item) in enumerate(bounded_samples(dataset, pending_indices, workers), 1):
        row = pending_rows[pending_row]
        if set(item) not in (FIELDS, FIELDS | {"metadata"}):
            raise ValueError(f"Unsupported packed sample fields: {set(item)}")
        if item["input_ids"].dtype != torch.long or str(item["doc_lens"].dtype) not in DOC_DTYPES:
            raise ValueError("Expected integer token IDs and document lengths")
        if item["label_mask"].dtype != torch.bool:
            raise ValueError("Expected boolean loss mask")
        ids = item["input_ids"].numpy()
        if ids.shape != shape[1:] or (ids < 0).any() or (ids >= dataset.vocab_size).any():
            raise ValueError("Invalid token shape or vocabulary ID")
        tokens[row] = ids
        masks[row] = item["label_mask"].numpy()
        restored = {
            "input_ids": torch.tensor(tokens[row].astype(np.int64)),
            "label_mask": torch.tensor(masks[row]),
            "doc_lens": torch.tensor(item["doc_lens"].tolist(), dtype=item["doc_lens"].dtype),
        }
        if any(not torch.equal(item[key], restored[key]) for key in FIELDS):
            raise ValueError("Packed sample changed during cache serialization")
        rows[row] = {
            "index": indices[row],
            "doc_lens": item["doc_lens"].tolist(),
            "doc_lens_dtype": str(item["doc_lens"].dtype),
            "hashes": {key: tensor_hash(item[key]) for key in sorted(FIELDS)},
        }
        if "metadata" in item:
            metadata = json.loads(json.dumps(item["metadata"]))
            if metadata != item["metadata"]:
                raise ValueError("Sample metadata changed during serialization")
            rows[row]["metadata"] = metadata
        if finished % checkpoint_interval == 0 or finished == len(pending_indices):
            save_progress()
            logger.info("Verified packed sample cache: %s/%s", completed + finished, len(indices))
    tokens.flush()
    masks.flush()
    files = {}
    for name in ("input_ids.npy", "label_mask.npy"):
        with (output / name).open("rb") as stream:
            files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {**identity, "files": files, "rows": rows}
    temporary = output / "manifest.partial.json"
    temporary.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest


class CachedPackedFSLDataset(NumpyPackedFSLDataset):
    """Read cached rows with per-sample integrity checks; retain native fallback."""

    def __init__(self, source, directory):
        self.__dict__.update(source.__dict__)
        self._native_source = source
        self._cache_directory = Path(directory).resolve()
        manifest_bytes = (self._cache_directory / "manifest.json").read_bytes()
        self._cache_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        self._open_cache(json.loads(manifest_bytes))

    def _open_cache(self, manifest):
        source, directory = self._native_source, self._cache_directory
        if (
            manifest["version"] != 1
            or manifest["dataset_fingerprint"] != source.fingerprint
            or manifest["dataset_instances"] != len(source)
            or manifest["sequence_length"] != source.sequence_length
            or manifest["source_paths"] != [str(path) for path in source.paths]
            or manifest["label_mask_paths"] != [str(path) for path in (source._label_mask_paths or [])]
        ):
            raise ValueError("Packed sample cache belongs to another dataset")
        self._cached_tokens = np.load(directory / "input_ids.npy", mmap_mode="r", allow_pickle=False)
        self._cached_masks = np.load(directory / "label_mask.npy", mmap_mode="r", allow_pickle=False)
        rows = manifest["rows"]
        shape = (len(rows), self.sequence_length)
        if (
            self._cached_tokens.shape != shape
            or self._cached_tokens.dtype != np.uint32
            or self._cached_masks.shape != shape
            or self._cached_masks.dtype != np.bool_
        ):
            raise ValueError("Invalid cached array shape or dtype")
        self._cached_rows = {entry["index"]: (row, entry) for row, entry in enumerate(rows)}
        if len(self._cached_rows) != len(rows):
            raise ValueError("Duplicate packed sample cache index")

    def __getstate__(self):
        # NumPy memmap pickling copies the complete array and restores an
        # anonymous allocation. Core spawns loader workers, so transmitting the
        # full cache would multiply its RAM and IPC cost by every worker/rank.
        state = self.__dict__.copy()
        for name in ("_cached_tokens", "_cached_masks", "_cached_rows"):
            state.pop(name)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        manifest_bytes = (self._cache_directory / "manifest.json").read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != self._cache_manifest_sha256:
            raise ValueError("Packed cache manifest changed during worker startup")
        self._open_cache(json.loads(manifest_bytes))

    @property
    def fingerprint(self):
        # Core includes the Python class name. This adapter preserves the original
        # data contents and must retain that native identity for checkpoint resume.
        return self._native_source.fingerprint

    def __getitem__(self, index):
        index = int(index)
        index = index if index >= 0 else len(self) + index
        if index not in self._cached_rows:
            # The normal loader may prefetch beyond the bounded training window.
            return super().__getitem__(index)
        row, entry = self._cached_rows[index]
        item = {
            "input_ids": torch.tensor(self._cached_tokens[row].astype(np.int64)),
            "label_mask": torch.tensor(self._cached_masks[row]),
            "doc_lens": torch.tensor(entry["doc_lens"], dtype=DOC_DTYPES[entry["doc_lens_dtype"]]),
        }
        if {key: tensor_hash(value) for key, value in item.items()} != entry["hashes"]:
            raise ValueError(f"Cached packed sample integrity failure: {index}")
        if "metadata" in entry:
            item["metadata"] = copy.deepcopy(entry["metadata"])
        return item
