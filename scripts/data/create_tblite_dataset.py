"""Create an HF dataset from the OpenThoughts-TBLite GitHub repo for Harbor GRPO training.

Each row maps a task to a local path convention (/tmp/tblite/{task_name}) and includes
the instruction as a chat message.  At training time the repo must be cloned to /tmp/tblite/
so the paths resolve.

Usage:
    uv run python scripts/data/create_tblite_dataset.py \
        --repo-url https://github.com/open-thoughts/OpenThoughts-TBLite.git \
        --push-to ai2-adapt-dev/openthoughts-tblite-harbor
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from datasets import Dataset


TASK_DIR_PREFIX = "/tmp/tblite"


def find_task_dirs(repo_root: Path) -> list[Path]:
    """Find directories that contain task.toml (i.e. Harbor task directories)."""
    return sorted(d.parent for d in repo_root.rglob("task.toml") if d.parent != repo_root)


def build_rows(task_dirs: list[Path]) -> list[dict]:
    rows = []
    for task_dir in task_dirs:
        task_name = task_dir.name
        instruction_path = task_dir / "instruction.md"
        instruction = instruction_path.read_text().strip() if instruction_path.exists() else "Solve the task."

        rows.append(
            {
                "task_name": task_name,
                "messages": [{"role": "user", "content": instruction}],
                "harbor_task_path": f"{TASK_DIR_PREFIX}/{task_name}",
                "ground_truth": ["harbor"],
                "dataset": ["harbor"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OpenThoughts-TBLite Harbor dataset")
    parser.add_argument(
        "--repo-url",
        default="https://github.com/open-thoughts/OpenThoughts-TBLite.git",
        help="Git URL for the OpenThoughts-TBLite repo",
    )
    parser.add_argument("--push-to", default="ai2-adapt-dev/openthoughts-tblite-harbor", help="HF repo to push to")
    parser.add_argument("--local-only", action="store_true", help="Don't push to HF, just print stats")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {args.repo_url} ...")
        subprocess.run(["git", "clone", "--depth", "1", args.repo_url, tmpdir], check=True)

        task_dirs = find_task_dirs(Path(tmpdir))
        print(f"Found {len(task_dirs)} task directories")

        rows = build_rows(task_dirs)
        ds = Dataset.from_list(rows)
        print(ds)
        print(f"\nSample row:\n{rows[0]}")

        if not args.local_only:
            print(f"\nPushing to {args.push_to} ...")
            ds.push_to_hub(args.push_to)
            print("Done.")


if __name__ == "__main__":
    main()
