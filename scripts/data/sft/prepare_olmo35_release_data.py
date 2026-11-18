"""Apply an audited exact-ID exclusion manifest to local release Parquet shards.

CPU only; reads bounded batches and publishes a new directory only after every
shard and the audit receipt pass validation. Input files are never modified.
"""

import argparse
import glob
import hashlib
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import parquet as pq


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_release_data(data_files, audit_manifest, output_dir, *, allow_missing_ids=False, batch_size=1024):
    """Return the receipt after atomically publishing the filtered shards."""
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise ValueError(f"Output already exists: {output_dir}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    paths = []
    for pattern in data_files:
        matches = sorted(glob.glob(str(pattern)))
        if not matches:
            raise ValueError(f"No input files match: {pattern}")
        paths.extend(Path(path).resolve(strict=True) for path in matches)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("Input files must be nonempty and contain no duplicate paths")

    audit_manifest = Path(audit_manifest)
    manifest_bytes = audit_manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    for key in ("source_dataset", "source_revision"):
        if not isinstance(manifest.get(key), str) or not manifest[key].strip():
            raise ValueError(f"Manifest requires a nonempty {key}")
    groups = manifest.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise ValueError("Manifest requires nonempty groups")
    if not isinstance(manifest.get("evidence"), list):
        raise ValueError("Manifest requires an evidence list")
    for name, ids in groups.items():
        if not isinstance(ids, list) or any(not isinstance(value, str) for value in ids):
            raise ValueError(f"Manifest group {name} must contain only string IDs")
    groups = {name: set(ids) for name, ids in groups.items()}
    requested = set().union(*groups.values())
    found = set()
    per_source = Counter()
    has_source = False
    receipt = {
        "audit_source": {"dataset": manifest["source_dataset"], "revision": manifest["source_revision"]},
        "audit_manifest": {
            "path": str(audit_manifest.resolve()),
            "sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        "allow_missing_ids": allow_missing_ids,
        "rows_in": 0,
        "rows_out": 0,
        "shards": [],
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output_dir.name}-", dir=output_dir.parent) as staging:
        staging = Path(staging)
        for index, path in enumerate(paths):
            source_hash = sha256(path)
            parquet = pq.ParquetFile(path)
            schema = parquet.schema_arrow
            id_index = schema.get_field_index("id")
            if id_index < 0 or not (
                pa.types.is_string(schema.field(id_index).type)
                or pa.types.is_large_string(schema.field(id_index).type)
            ):
                raise ValueError(f"{path}: requires exactly one string id column")
            if parquet.metadata.num_rows == 0:
                raise ValueError(f"Empty input shard: {path}")
            output_path = staging / f"{index:05d}-{path.name}"
            dropped = rows_in = 0
            source_index = schema.get_field_index("source")
            has_source |= source_index >= 0
            with pq.ParquetWriter(output_path, schema, use_compliant_nested_type=False) as writer:
                value_set = pa.array(sorted(requested), type=schema.field(id_index).type)
                for batch in parquet.iter_batches(batch_size=batch_size):
                    ids = batch.column(id_index)
                    if ids.null_count:
                        raise ValueError(f"{path}: null id values are not allowed")
                    exclude = pc.is_in(ids, value_set=value_set)
                    dropped_batch = batch.filter(exclude)
                    found.update(dropped_batch.column(id_index).to_pylist())
                    if source_index >= 0:
                        per_source.update(
                            json.dumps(value, sort_keys=True)
                            for value in dropped_batch.column(source_index).to_pylist()
                        )
                    rows_in += batch.num_rows
                    dropped += dropped_batch.num_rows
                    writer.write_batch(batch.filter(pc.invert(exclude)))
            if sha256(path) != source_hash:
                raise ValueError(f"Input changed while processing: {path}")
            receipt["shards"].append(
                {
                    "input": str(path),
                    "input_sha256": source_hash,
                    "output": output_path.name,
                    "output_sha256": sha256(output_path),
                    "rows_in": rows_in,
                    "rows_out": rows_in - dropped,
                    "rows_dropped": dropped,
                }
            )
            receipt["rows_in"] += rows_in
            receipt["rows_out"] += rows_in - dropped
        missing = requested - found
        if missing and not allow_missing_ids:
            raise ValueError(
                f"{len(missing)} requested IDs absent from inputs; use --allow-missing-ids only for already-filtered inputs"
            )
        receipt["rows_dropped"] = receipt["rows_in"] - receipt["rows_out"]
        receipt["matched_unique_ids"] = len(found)
        receipt["groups"] = {
            name: {
                "requested": len(ids),
                "matched": len(ids & found),
                "absent": len(ids - found),
                "absent_ids": sorted(ids - found),
            }
            for name, ids in groups.items()
        }
        if has_source:
            receipt["per_source_dropped"] = [
                {"source": json.loads(source), "rows_dropped": count} for source, count in sorted(per_source.items())
            ]
        (staging / "receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
        if output_dir.exists() or output_dir.is_symlink():
            raise ValueError(f"Output already exists: {output_dir}")
        staging.rename(output_dir)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-files", nargs="+", required=True, help="Local Parquet paths or quoted glob patterns")
    parser.add_argument("--audit-manifest", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory for filtered shards and receipt.json")
    parser.add_argument(
        "--allow-missing-ids",
        action="store_true",
        help="Explicitly permit prior exclusions in already-filtered C12 inputs",
    )
    args = parser.parse_args()
    try:
        receipt = prepare_release_data(
            args.data_files, args.audit_manifest, args.output_dir, allow_missing_ids=args.allow_missing_ids
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({key: receipt[key] for key in ("rows_in", "rows_out", "rows_dropped")}))


if __name__ == "__main__":
    main()
