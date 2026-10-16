"""Pin CPU-resolved SFT dataset metadata without reopening raw Arrow data on GPUs."""

import hashlib
import json
from pathlib import Path

from open_instruct import dataset_transformation


def write(path, request, dcs, tokenizer_config):
    records = [dataset_transformation._get_serializable_dataset_config_dict(dc, exclude_none=True) for dc in dcs]
    manifest = {
        "schema": 1,
        "request": request,
        "dataset_configs": records,
        "cache_hash": dataset_transformation.compute_config_hash(dcs, tokenizer_config),
        "cache_version": dataset_transformation.DATASET_CACHE_VERSION,
    }
    # Ensure the serialized path reproduces the actual historical resolver's hash.
    if dataset_transformation.compute_config_hash_from_dicts(records, tokenizer_config) != manifest["cache_hash"]:
        raise ValueError("Resolved manifest does not reproduce the dataset cache hash")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, sort_keys=True, indent=2) + "\n"
    if path.exists() and path.read_text() != payload:
        raise FileExistsError(f"Refusing to replace different resolved dataset metadata: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def read(path, request, tokenizer_config):
    payload = Path(path).read_bytes()
    manifest = json.loads(payload)
    if manifest.get("schema") != 1 or manifest.get("cache_version") != dataset_transformation.DATASET_CACHE_VERSION:
        raise ValueError("Unsupported resolved dataset manifest version")
    if manifest["request"] != request:
        raise ValueError("Resolved dataset request differs from training arguments")
    cache_hash = dataset_transformation.compute_config_hash_from_dicts(manifest["dataset_configs"], tokenizer_config)
    if cache_hash != manifest["cache_hash"]:
        raise ValueError("Resolved dataset metadata or current tokenizer changed its cache hash")
    return cache_hash, hashlib.sha256(payload).hexdigest()
