"""Retain completed Megatron saves for diagnosis without changing training retention."""

import argparse
import json
import os
import time
from pathlib import Path


def retain(root, destination, iteration):
    root, destination = Path(root), Path(destination)
    source = root / f"iter_{iteration:07d}"
    target = destination / source.name
    marker = target / "diagnostic-retention.json"
    if marker.exists():
        record = json.loads(marker.read_text())
        if record["source"] != str(source) or record["iteration"] != iteration:
            raise ValueError("Existing diagnostic retention has different provenance")
        return record
    tracker = root / "latest_checkpointed_iteration.txt"
    cursor = root / "rollout" / f"global_dataset_state_dict_{iteration}.pt"
    if not tracker.is_file() or int(tracker.read_text().strip()) < iteration or not cursor.is_file():
        return None
    if not source.is_dir():
        raise FileNotFoundError(f"Completed iteration was pruned before retention: {source}")
    files = sorted(path for path in source.rglob("*") if not path.is_dir())
    if not files or not any(path.name == ".metadata" for path in files):
        raise ValueError(f"Missing completed distributed checkpoint metadata: {source}")
    # Saves have unique iteration directories. The native synchronous save writes its
    # tracker only after completion; the separate rollout cursor completes afterward.
    # Hard links preserve bytes through later unlink-based native retention without
    # modifying files, replacing the tracker, or copying hundreds of GB over WEKA.
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")) or cursor.is_symlink():
        raise ValueError("Diagnostic retention requires ordinary files and directories")
    destination.mkdir(parents=True, exist_ok=True)
    target.mkdir(exist_ok=True)
    records = []
    for path, relative in [(p, p.relative_to(source)) for p in files] + [(cursor, Path("rollout") / cursor.name)]:
        saved = target / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        before = path.stat()
        if saved.exists():
            if not os.path.samefile(path, saved):
                raise ValueError(f"Refusing to replace unrelated diagnostic file: {saved}")
        else:
            os.link(path, saved)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Checkpoint changed during retention: {path}")
        records.append({"path": str(relative), "bytes": after.st_size, "mtime_ns": after.st_mtime_ns})
    record = {
        "source": str(source),
        "iteration": iteration,
        "completed_updates": iteration + 1,
        "method": "hard links to completed immutable iteration files; original save path and pruning unchanged",
        "retained_bytes": sum(item["bytes"] for item in records),
        "files": records,
        "scope": "Completed-save retention only; tensor restoration and parameter drift are separate checks.",
    }
    temporary = marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(marker)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-hours", type=float, default=18)
    args = parser.parse_args()
    if args.destination.resolve() == args.root.resolve() or args.root.resolve() in args.destination.resolve().parents:
        raise ValueError("Diagnostic retention must be outside the trainer checkpoint tree")
    deadline = time.monotonic() + args.max_hours * 3600
    # Final499 remains the trainer's latest; preserve only intermediate saves.
    pending = {99, 199, 299, 399}
    while pending and time.monotonic() < deadline:
        for iteration in sorted(pending):
            # The next rollout dump proves the synchronous prior save and its
            # dataset cursor write returned before we record file inventories.
            if not (args.root.parent / "rollout_data" / f"{iteration + 1}.pt").is_file():
                continue
            record = retain(args.root, args.destination, iteration)
            if record:
                print(json.dumps({k: v for k, v in record.items() if k != "files"}), flush=True)
                pending.remove(iteration)
        if pending:
            time.sleep(60)
    if pending:
        raise TimeoutError(f"Save boundaries not observed before deadline: {sorted(pending)}")


if __name__ == "__main__":
    main()
