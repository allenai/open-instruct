"""Globally shuffle paired full corpora with bounded memory and exact provenance."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
from pyarrow import compute as pc
from pyarrow import parquet as pq


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(raw, uniform, out, shards, seed, expected_rows, expected_raw=None, expected_uniform=None):
    out.mkdir(exist_ok=False)
    files = sorted(raw.glob("*.parquet"))
    assert files and [p.name for p in files] == [p.name for p in sorted(uniform.glob("*.parquet"))]
    sizes = [pq.read_metadata(p).num_rows for p in files]
    total = sum(sizes)
    assert total == expected_rows and 0 < shards <= total
    order = np.random.default_rng(seed).permutation(total)
    rank = np.empty(total, dtype=np.int64)
    rank[order] = np.arange(total)
    np.save(out / "source-row-indices.npy", order)
    writers = {}
    sources = Counter()
    changed = 0
    offset = 0
    input_receipts = []
    for arm in ("control", "uniform"):
        (out / arm).mkdir()
        (out / f"{arm}-unsorted").mkdir()
    try:
        for file, size in zip(files, sizes, strict=True):
            other = uniform / file.name
            input_hashes = {"control": digest(file), "uniform": digest(other)}
            if expected_raw is not None:
                assert input_hashes["control"] == expected_raw[file.name]
                assert input_hashes["uniform"] == expected_uniform[file.name]
            input_receipts.append({"file": file.name, "rows": size, "sha256": input_hashes})
            tables = {"control": pq.read_table(file), "uniform": pq.read_table(other)}
            a, b = tables.values()
            assert a.schema == b.schema and len(a) == len(b) == size
            for column in a.column_names:
                if column != "messages":
                    assert a[column].equals(b[column]), column
            for begin in range(0, size, 256):
                am = a["messages"].slice(begin, 256).to_pylist()
                bm = b["messages"].slice(begin, 256).to_pylist()
                for x, y in zip(am, bm, strict=True):
                    assert len(x) == len(y)
                    for xm, ym in zip(x, y, strict=True):
                        assert xm["role"] == ym["role"]
                        if xm["role"] != "user":
                            assert xm == ym, "Non-user message changed"
                    changed += int(x != y)
            sources.update(a["source"].to_pylist())
            ranks = rank[offset : offset + size]
            buckets = ranks * shards // total
            for arm, table in tables.items():
                table = table.append_column("_shuffle_rank", pa.array(ranks))
                for bucket in np.unique(buckets):
                    bucket = int(bucket)
                    chunk = table.take(pa.array(np.flatnonzero(buckets == bucket)))
                    key = (arm, bucket)
                    if key not in writers:
                        path = out / f"{arm}-unsorted" / f"{bucket:05d}.parquet"
                        writers[key] = pq.ParquetWriter(path, chunk.schema, compression="zstd")
                    writers[key].write_table(chunk)
            offset += size
            print("PARTITIONED", file.name, offset, "of", total, flush=True)
    finally:
        for writer in writers.values():
            writer.close()
    assert offset == total
    output_files = {}
    for bucket in range(shards):
        ids = None
        for arm in ("control", "uniform"):
            staged = out / f"{arm}-unsorted" / f"{bucket:05d}.parquet"
            table = pq.read_table(staged)
            table = table.take(pc.sort_indices(table, sort_keys=[("_shuffle_rank", "ascending")]))
            ranks = table["_shuffle_rank"].to_numpy()
            lo, hi = (bucket * total + shards - 1) // shards, ((bucket + 1) * total + shards - 1) // shards
            assert np.array_equal(ranks, np.arange(lo, hi))
            if ids is None:
                ids = table["id"]
            else:
                assert ids.equals(table["id"])
            table = table.drop(["_shuffle_rank"])
            path = out / arm / f"train-{bucket:05d}-of-{shards:05d}.parquet"
            pq.write_table(table, path, compression="zstd", row_group_size=256)
            output_files[str(path.relative_to(out))] = {
                "rows": len(table),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            # Only this script's newly-created disposable shuffle intermediates are removed.
            staged.unlink()
        if bucket % 32 == 0:
            print("SHARD-COMPLETE", bucket, "of", shards, flush=True)
    for arm in ("control", "uniform"):
        (out / f"{arm}-unsorted").rmdir()
    receipt = {
        "source_total": total,
        "rows_per_arm": total,
        "seed": seed,
        "shards": shards,
        "changed_records": changed,
        "source_counts": dict(sources),
        "non_user_messages_preserved_every_record": True,
        "paired_prepacking_order_identical": True,
        "source_order_sha256": digest(out / "source-row-indices.npy"),
        "inputs": input_receipts,
        "files": output_files,
        "scope": "Full H127 corpora; not the infrastructure smoke subset",
        "caveat": "Prompt lengths differ, so packed token boundaries and exposure at fixed tokens can differ",
    }
    (out / "manifest.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/storage/dolci-think-sft"))
    parser.add_argument("--out", type=Path, default=Path("/storage/dolci-think-sft/tpu-presentation14t-full-v1"))
    args = parser.parse_args()
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    source_manifest = json.loads((args.root / "presentation-structured-v1/manifest.json").read_text())
    treatment_manifest = json.loads(
        (args.root / "presentation-structured-v1/uniform-balanced/build-manifest.json").read_text()
    )
    receipt = prepare(
        args.root / "untokenized-hf/data",
        args.root / "presentation-structured-v1/uniform-balanced",
        args.out,
        shards=512,
        seed=0,
        expected_rows=2253684,
        expected_raw={r["shard"]: r["raw_sha256"] for r in source_manifest["shards"]},
        expected_uniform={r["file"]: r["sha256"] for r in treatment_manifest["files"]},
    )
    assert receipt["changed_records"] == 2008662
    shutil.copytree(args.root / "tpu-presentation14t-smoke-v1/tokenizer", args.out / "tokenizer")
    receipt["tokenizer_sha256"] = {p.name: digest(p) for p in (args.out / "tokenizer").iterdir()}
    (args.out / "manifest.json").write_text(json.dumps(receipt, indent=2))
    print("FULL-PAIR-READY", receipt["rows_per_arm"], receipt["changed_records"], flush=True)


if __name__ == "__main__":
    main()
