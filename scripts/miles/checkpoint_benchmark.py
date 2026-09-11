"""Profile native saves and compare fresh resume against the same saved trajectory."""

import argparse
import json
from pathlib import Path

from scripts.miles import durable_continuation as continuation

from open_instruct.miles.state import atomic_json

MODES = {
    "baseline": {"profile": True},
    "compact": {"profile": True, "compact_storage": True},
    "balanced": {"profile": True, "compact_storage": True, "dedup_save_to_lowest_rank": False},
    "metadata": {
        "profile": True,
        "compact_storage": True,
        "dedup_save_to_lowest_rank": False,
        "constant_memory_planning": True,
    },
    "processes": {
        "profile": True,
        "compact_storage": True,
        "dedup_save_to_lowest_rank": False,
        "process_count": 2,
        "thread_count": 2,
    },
}

MODES["metadata_processes"] = {**MODES["metadata"], "process_count": 2, "thread_count": 2}


def audit(root, world):
    failures, timings = [], []
    for rank in range(world):
        saved = json.loads((root / "split" / f"rank{rank}.json").read_text())
        resumed = json.loads((root / "resumed" / f"rank{rank}.json").read_text())
        if saved["pid"] == resumed["pid"]:
            failures.append(f"rank{rank}: resume must run in a fresh process")
        for field in ("runtime_lock", "harness_sha256", "scheduler_horizon", "world", "rank"):
            if saved[field] != resumed[field]:
                failures.append(f"rank{rank}: {field} changed")
        for left, right in (
            (saved["snapshots"]["step2"], saved["snapshots"]["after_save2"]),
            (saved["snapshots"]["step2"], resumed["snapshots"]["restored2"]),
            (saved["snapshots"]["step4"], resumed["snapshots"]["step4"]),
        ):
            failures.extend(f"rank{rank}: {item}" for item in continuation.compare_states(left, right))
        if saved["world"] != world or saved["rank"] != rank:
            failures.append(f"rank{rank}: unexpected measurement topology")
        for suffix in (".main", ".exp_avg", ".exp_avg_sq"):
            first = saved["snapshots"]["step2"]["optimizer"]
            last = saved["snapshots"]["step4"]["optimizer"]
            names = [name for name in first if name.endswith(suffix)]
            if not names or not any(first[name] != last.get(name) for name in names):
                failures.append(f"rank{rank}: no continuing {suffix} update signal")
        for step in ("3", "4"):
            if not saved["score_hashes"].get(step) or saved["score_hashes"][step] != resumed["score_hashes"].get(step):
                failures.append(f"rank{rank}: step {step} scoring log-probabilities differ")
        if not saved["exports"].get("boundary") or saved["exports"] != resumed["exports"]:
            failures.append(f"rank{rank}: HF export differs across resume")
        if len(saved["save_timings"]) != 1:
            failures.append(f"rank{rank}: expected exactly one native save")
        timings.extend({"rank": rank, **value} for value in saved["save_timings"])
    report = {
        "correctness_passed": not failures,
        "failures": failures,
        "save_timings": timings,
        "max_save_seconds": max(value["total_seconds"] for value in timings) if timings else None,
        "performance_target_seconds": 340,
        "limits": ["Fixed synthetic responses, same EP topology; no SGLang transport or decode tested"],
    }
    report["performance_passed"] = bool(timings) and report["max_save_seconds"] <= 340
    atomic_json(root / "checkpoint-audit.json", report)
    print("CHECKPOINT_AUDIT", json.dumps(report), flush=True)
    if failures:
        raise ValueError("Checkpoint correctness gate failed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "audit"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--hf", type=Path)
    parser.add_argument("--phase", choices=("split", "resumed"))
    parser.add_argument("--mode", choices=MODES, default="baseline")
    parser.add_argument("--backend", choices=("torch", "flash_4"), default="torch")
    parser.add_argument("--world", type=int, default=1)
    args = parser.parse_args()
    if args.command == "prepare":
        continuation.prepare(args.root, args.hf)
    elif args.command == "audit":
        audit(args.root, args.world)
    else:
        if args.phase is None:
            parser.error("run requires --phase")
        continuation.run(
            args.root,
            args.phase,
            args.backend,
            checkpoint_options=MODES[args.mode],
            continue_after_save=True,
            verify_export=True,
        )


if __name__ == "__main__":
    main()
