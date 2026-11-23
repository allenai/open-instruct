"""Compact, hash-bound offsets for prepared JSONL; never retain the row population."""

import fcntl
import hashlib
import json
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

DTYPE = np.dtype([("offset", "<u8"), ("length", "<u8"), ("pool", "<u4")])


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def open_index(path, expected_hash, expected_rows, max_prompt_length=None):
    """Build once under a file lock, then memory-map offsets; verify immutable inputs.

    Old prepared bundles need no migration. Sidecars are derived caches, and do
    not change prompt bytes or manifest identity. A failed build leaves no ready
    header, so readers cannot consume partial indexes.
    """
    path = Path(path)
    prefix = path.with_name(f"{path.name}.{expected_hash[:16]}.index")
    header_path, index_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".bin")
    with Path(str(prefix) + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not header_path.exists():
            pools, count, max_tokens, offset = {}, 0, 0, 0
            source_digest = hashlib.sha256()
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".row-index-", delete=False) as output:
                temporary = output.name
                with path.open("rb") as source:
                    for line in source:
                        source_digest.update(line)
                        if not line.strip():
                            offset += len(line)
                            continue
                        row = json.loads(line)
                        metadata = row.get("metadata") or {}
                        pool, tokens = metadata.get("sampling_pool"), metadata.get("run_prompt_tokens")
                        if not isinstance(row.get("input"), str) or not isinstance(pool, str):
                            raise ValueError("Indexed pools require prepared text and sampling_pool metadata")
                        if type(tokens) is not int or tokens < 1:
                            raise ValueError("Indexed pools require positive prepared prompt lengths")
                        code = pools.setdefault(pool, len(pools))
                        output.write(struct.pack("<QQI", offset, len(line), code))
                        offset += len(line)
                        count += 1
                        max_tokens = max(max_tokens, tokens)
                output.flush()
                os.fsync(output.fileno())
            if source_digest.hexdigest() != expected_hash or count != expected_rows:
                raise ValueError("Prepared source hash or row count changed")
            os.replace(temporary, index_path)
            header = {
                "version": 1,
                "source_sha256": expected_hash,
                "rows": count,
                "max_prompt_tokens": max_tokens,
                "pools": list(pools),
                "index_sha256": digest(index_path),
            }
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".row-header-", delete=False) as out:
                json.dump(header, out)
                out.flush()
                os.fsync(out.fileno())
                temporary = out.name
            os.replace(temporary, header_path)
        else:
            if digest(path) != expected_hash:
                raise ValueError("Prepared source hash changed")
        header = json.loads(header_path.read_text())
        if (header.get("version"), header.get("source_sha256"), header.get("rows")) != (
            1,
            expected_hash,
            expected_rows,
        ):
            raise ValueError("Prepared row index identity mismatch")
        if index_path.stat().st_size != expected_rows * DTYPE.itemsize or digest(index_path) != header["index_sha256"]:
            raise ValueError("Prepared row index is incomplete or changed")
        if max_prompt_length and header["max_prompt_tokens"] > max_prompt_length:
            raise ValueError("Prepared source exceeds prompt length limit")
    rows = np.memmap(index_path, dtype=DTYPE, mode="r", shape=(expected_rows,))
    return rows, header
