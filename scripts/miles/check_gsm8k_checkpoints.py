"""Check completed native save boundaries without loading large optimizer tensors."""

import hashlib
import json
from pathlib import Path


def check_boundaries(root, *, updates, save_interval, world_size=2, expert_parallel_size=2):
    root = Path(root)
    if save_interval <= 0 or updates <= 0 or updates % save_interval:
        raise ValueError("Save interval must divide the positive update horizon")
    records = []
    for step in range(save_interval, updates + 1, save_interval):
        rollout = step - 1
        path = root / "core" / f"rollout_{rollout:07d}"
        raw = (path / "complete.json").read_bytes()
        manifest = json.loads(raw)
        expected = {"schema_version": 2, "world_size": world_size, "expert_parallel_size": expert_parallel_size}
        if any(type(manifest.get(key)) is not int or manifest[key] != value for key, value in expected.items()):
            raise ValueError(f"Unexpected checkpoint topology/schema at update{step}")
        clock = manifest["clock"]
        if any(
            type(clock.get(key)) is not int or clock[key] != value
            for key, value in {"completed_steps": step, "next_rollout_id": step}.items()
        ):
            raise ValueError(f"Checkpoint clock differs at update{step}")
        cursor = root / "rollout" / f"global_dataset_state_dict_{rollout}.pt"
        cursor_hash = hashlib.sha256(cursor.read_bytes()).hexdigest()
        if cursor_hash != manifest["cursor_sha256"]:
            raise ValueError(f"Checkpoint cursor differs at update{step}")
        files = [path / "model/.metadata", *[path / f"rank_{rank}.pt" for rank in range(world_size)]]
        shards = list((path / "model").glob("*.distcp"))
        if not shards or any(not item.is_file() or item.stat().st_size == 0 for item in files + shards):
            raise ValueError(f"Missing native checkpoint files at update{step}")
        records.append(
            {
                "completed_steps": step,
                "rollout_id": rollout,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "cursor_sha256": cursor_hash,
                "native_shards": len(shards),
                "native_bytes": sum(item.stat().st_size for item in shards),
            }
        )
    latest = json.loads((root / "core-latest.json").read_text())
    if type(latest.get("rollout_id")) is not int or latest["rollout_id"] != updates - 1:
        raise ValueError("Latest checkpoint does not identify final update")
    return {
        "committed_boundaries": records,
        "scope": "Manifest/clock/topology/cursor digest and nonempty native-file inventory; full tensor restoration is qualified separately.",
    }
