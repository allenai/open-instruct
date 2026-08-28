"""Build the repo-local rl_rollouts/ symlink farm from the training registry.

The shared rollout dump on Weka (e.g. /weka/oe-adapt-default/allennlp/deletable_rollouts)
holds hundreds of thousands of files, so pointing the viewer's RolloutStore at it
directly makes the startup rglob scan take forever. Instead, each registry entry
declares its rollout source directory and attempt prefixes, and this tool creates
symlinks for just those attempts' metadata and shard files under
<repo>/rl_rollouts/<training-id>/. Nothing is copied or moved.

Registry entries opt in with a per-rollout `source` key naming the real directory,
while `path` points at the repo-local farm directory this tool maintains:

    rollouts:
    - path: rl_rollouts/my-training-id
      source: /weka/oe-adapt-default/allennlp/deletable_rollouts
      attempts:
      - my_exp__42__1784051570

Re-run after registering new attempts or when a live run grows new shards
(new shard files need new symlinks; existing symlinks track file growth for free).

Usage:
    uv run --no-sync python -m viewer.tools.link_rollouts [--registry viewer/registry]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


_SOURCE_CACHE: dict[Path, list[str]] = {}


def source_files(source: Path) -> list[str]:
    """One flat scandir per source directory; shared dumps hold ~200k entries."""
    cached = _SOURCE_CACHE.get(source)
    if cached is None:
        with os.scandir(source) as entries:
            cached = [entry.name for entry in entries if entry.name.endswith(".jsonl")]
        _SOURCE_CACHE[source] = cached
    return cached


def link_attempt(source: Path, farm: Path, attempt: str) -> tuple[int, int]:
    created = existing = 0
    for name in source_files(source):
        if not name.startswith(attempt):
            continue
        suffix = name[len(attempt) :]
        if not (suffix.startswith("_metadata") or "_rollouts_" in suffix or "_filtered_rollouts_" in suffix):
            continue  # skip trainer-logprob shards and other artifacts the viewer never reads
        target = farm / name
        if target.is_symlink() or target.exists():
            existing += 1
            continue
        target.symlink_to(source / name)
        created += 1
    return created, existing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--registry", default=str(REPO_ROOT / "viewer" / "registry"))
    args = parser.parse_args()

    trainings_dir = Path(args.registry) / "trainings"
    for source_file in sorted(trainings_dir.glob("*.yaml")):
        payload = yaml.safe_load(source_file.read_text())
        for launch in payload.get("launches") or []:
            for rollout in launch.get("rollouts") or []:
                source = rollout.get("source")
                attempts = rollout.get("attempts") or []
                if not source or not attempts:
                    continue
                source_path = Path(source)
                farm = (REPO_ROOT / rollout["path"]).resolve()
                farm.mkdir(parents=True, exist_ok=True)
                for attempt in attempts:
                    created, existing = link_attempt(source_path, farm, attempt)
                    print(f"{payload['id']}/{launch.get('id')}: {attempt}: +{created} linked, {existing} existing")


if __name__ == "__main__":
    main()
