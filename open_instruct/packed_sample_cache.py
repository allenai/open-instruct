"""Cache exact packed samples without changing dataset identity or loader order."""

import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from olmo_core.data import NumpyPackedFSLDataset

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)
FIELDS = {"input_ids", "label_mask", "doc_lens"}
DOC_DTYPES = {"torch.int32": torch.int32, "torch.int64": torch.int64}


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


def build_cache(dataset, indices, output, *, provenance, workers=8):
    """Materialize a bounded selection; publish the manifest only after verification."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    indices = [int(index) for index in indices]
    if not indices or len(set(indices)) != len(indices) or workers < 1:
        raise ValueError("Require unique nonempty indices and positive worker count")
    if any(index < 0 or index >= len(dataset) for index in indices):
        raise ValueError("Sample index outside source dataset")
    # Materialize lazy metadata once before sharing this read-only dataset with threads.
    _ = dataset.source_size_groups, dataset.source_instance_offsets
    shape = (len(indices), dataset.sequence_length)
    tokens = np.lib.format.open_memmap(output / "input_ids.npy", mode="w+", dtype=np.uint32, shape=shape)
    masks = np.lib.format.open_memmap(output / "label_mask.npy", mode="w+", dtype=np.bool_, shape=shape)
    rows = [None] * len(indices)
    for finished, (row, item) in enumerate(bounded_samples(dataset, indices, workers), 1):
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
        if finished % 128 == 0 or finished == len(indices):
            logger.info("Verified packed sample cache: %s/%s", finished, len(indices))
    tokens.flush()
    masks.flush()
    files = {}
    for name in ("input_ids.npy", "label_mask.npy"):
        with (output / name).open("rb") as stream:
            files[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest = {
        "version": 1,
        "dataset_fingerprint": dataset.fingerprint,
        "dataset_instances": len(dataset),
        "sequence_length": dataset.sequence_length,
        "source_paths": [str(path) for path in dataset.paths],
        "label_mask_paths": [str(path) for path in (dataset._label_mask_paths or [])],
        "provenance": provenance,
        "files": files,
        "rows": rows,
    }
    temporary = output / "manifest.partial.json"
    temporary.write_text(json.dumps(manifest, separators=(",", ":")) + "\n")
    temporary.replace(output / "manifest.json")
    return manifest


class CachedPackedFSLDataset(NumpyPackedFSLDataset):
    """Read cached rows with per-sample integrity checks; retain native fallback."""

    def __init__(self, source, directory):
        self.__dict__.update(source.__dict__)
        self._native_source = source
        directory = Path(directory)
        manifest = json.loads((directory / "manifest.json").read_text())
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
