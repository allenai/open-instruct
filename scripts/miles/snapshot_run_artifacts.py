"""Losslessly snapshot a finished run, including tensors and large logs.

Run this only after the producing attempt is terminal. Symlinks are inventoried,
not followed: prepared checkpoint links must not copy external model weights.
"""

import argparse
import hashlib
import json
import os
import stat
import tarfile
from datetime import UTC, datetime
from pathlib import Path


def inventory(source):
    entries = []
    for path in sorted(source.rglob("*")):
        info = path.lstat()
        row = {
            "path": str(path.relative_to(source)),
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "inode": info.st_ino,
            "device": info.st_dev,
        }
        if stat.S_ISLNK(info.st_mode):
            row.update(kind="symlink", target=os.readlink(path))
        elif stat.S_ISREG(info.st_mode):
            row["kind"] = "file"
        elif stat.S_ISDIR(info.st_mode):
            row["kind"] = "directory"
        else:
            raise ValueError(f"Unsupported special file in run artifacts: {path}")
        entries.append(row)
    return entries


class HashReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data


def snapshot(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents:
        raise ValueError("Snapshot output must be outside the source tree")
    if not source.is_dir():
        raise ValueError("Snapshot source must be an existing run directory")
    output.mkdir(parents=True, exist_ok=False)
    before = inventory(source)
    files, links = [], []
    archive = output / "artifacts.tar.gz.partial"
    try:
        with tarfile.open(archive, "w:gz", compresslevel=1) as tar:
            for row in before:
                path = source / row["path"]
                if row["kind"] == "symlink":
                    links.append({**row, "action": "recorded_only_not_followed_or_materialized"})
                    continue
                info = tar.gettarinfo(str(path), arcname="run/" + row["path"])
                if row["kind"] == "directory":
                    tar.addfile(info)
                    continue
                if not info.isfile() or info.size != row["size"]:
                    raise ValueError(f"Source changed before copying: {path}")
                with path.open("rb") as stream:
                    reader = HashReader(stream)
                    tar.addfile(info, reader)
                if reader.bytes_read != row["size"]:
                    raise ValueError(f"Incomplete artifact read: {path}")
                files.append({**row, "sha256": reader.digest.hexdigest()})
        if inventory(source) != before:
            raise ValueError(
                "Source tree changed during snapshot; wait for a terminal producer and use a fresh output"
            )
        complete = output / "artifacts.tar.gz"
        archive.rename(complete)
        with complete.open("rb") as stream:
            archive_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest = {
            "status": "complete",
            "created_utc": datetime.now(UTC).isoformat(),
            "source": str(source),
            "archive": {"path": complete.name, "bytes": complete.stat().st_size, "sha256": archive_hash},
            "files": files,
            "file_count": len(files),
            "uncompressed_bytes": sum(row["size"] for row in files),
            "symlinks": links,
            "scope": "All regular files, without size or extension exclusions. Symlinks recorded without copying targets. Metadata inventory stable across copying; no claim of an atomic live-filesystem snapshot.",
        }
        (output / "snapshot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        return manifest
    except BaseException as error:
        (output / "snapshot-failure.json").write_text(
            json.dumps({"status": "failed", "type": type(error).__name__, "error": str(error)}, indent=2) + "\n"
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = snapshot(args.source, args.output)
    print(json.dumps({key: result[key] for key in ("status", "file_count", "uncompressed_bytes", "archive")}))


if __name__ == "__main__":
    main()
