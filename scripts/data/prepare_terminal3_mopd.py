"""Prepare data for the 3-dataset terminal MOPD experiments (local or in a Beaker job).

Two artifacts, both idempotent:

1. ``--task-data-dir``: one merged task-data directory symlinking every task dir
   from the three repos' extracted ``task-data.tar.gz`` trees (task ids are
   disjoint across the three). Needed because one swerl tool pool has a single
   task-data source, while a mixed run needs lookup across all three repos.
   Pass this path as ``tool_configs.task_data_dir``.

2. ``--jsonl-dir`` (optional): per-dataset jsonls with the ``dataset`` tag
   rewritten from the shared ``"passthrough"`` to ``termigen``/``endless``/
   ``tmax`` so MOPD ``route`` can tell the domains apart. Re-tagged rows skip
   the passthrough env reward, which is fine under ``--opd_pure``. Point
   ``--dataset_mixer_list`` at ``<jsonl-dir>/{termigen,endless,tmax}.jsonl``.

Example (Beaker payload, before grpo_fast.py):
    python scripts/data/prepare_terminal3_mopd.py \
        --task-data-dir /tmp/terminal3_task_data --jsonl-dir /tmp/mopd_terminal3
"""

import argparse
import os

from datasets import load_dataset

from open_instruct import logger_utils
from open_instruct.environments.swerl_sandbox import SWERLSandboxEnv

logger = logger_utils.setup_logger(__name__)

REPOS_AND_TAGS = [
    ("allenai/open-instruct-termigen", "termigen"),
    ("allenai/open-instruct-endless-terminals", "endless"),
    ("allenai/tmax-15k-open-instruct", "tmax"),
]


def merge_task_data(task_data_dir: str) -> None:
    marker = os.path.join(task_data_dir, ".merge_complete")
    if os.path.isfile(marker):
        logger.info(f"Merged task data already present at {task_data_dir}")
        return
    os.makedirs(task_data_dir, exist_ok=True)
    count = 0
    for repo, _ in REPOS_AND_TAGS:
        tree = SWERLSandboxEnv.resolve_task_data_dir(repo)
        for name in os.listdir(tree):
            src = os.path.join(tree, name)
            if not os.path.isdir(src):
                continue
            dst = os.path.join(task_data_dir, name)
            if os.path.islink(dst):
                os.remove(dst)
            os.symlink(src, dst)
            count += 1
    with open(marker, "w", encoding="utf-8") as f:
        f.write("ok\n")
    logger.info(f"Merged {count} task dirs into {task_data_dir}")


def write_retagged_jsonls(jsonl_dir: str, rows_per_dataset: int) -> None:
    os.makedirs(jsonl_dir, exist_ok=True)
    for repo, tag in REPOS_AND_TAGS:
        out = os.path.join(jsonl_dir, f"{tag}.jsonl")
        if os.path.isfile(out):
            logger.info(f"{out} already present")
            continue
        d = load_dataset(repo, split="train")
        if rows_per_dataset > 0:
            d = d.select(range(min(rows_per_dataset, len(d))))
        d = d.map(lambda ex, tag=tag: {"dataset": tag})
        d.to_json(out)
        logger.info(f"Wrote {len(d)} rows to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-data-dir", required=True, help="Output dir for the merged task data.")
    parser.add_argument("--jsonl-dir", default=None, help="Optional output dir for re-tagged jsonls (MOPD route).")
    parser.add_argument(
        "--rows-per-dataset",
        type=int,
        default=0,
        help="Cap rows per re-tagged jsonl (0 = all rows; use a small cap for smoke tests).",
    )
    args = parser.parse_args()
    merge_task_data(args.task_data_dir)
    if args.jsonl_dir:
        write_retagged_jsonls(args.jsonl_dir, args.rows_per_dataset)


if __name__ == "__main__":
    main()
