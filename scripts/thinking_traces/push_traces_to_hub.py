"""Publish generated thinking traces to the HuggingFace Hub.

Deliberately standalone and re-runnable. The generation job calls it as a
best-effort final step, but a run whose push failed -- no token, a transient Hub
error -- loses nothing: the traces are already durable in the Beaker result
dataset and on weka, and this script can be pointed at those files later to
backfill. That separation is the whole reason it is not inlined into the
generator, where a missing token would fail a job after hours of inference.

Each row keeps the keys needed to join back to the source corpus:
``dataset`` + ``dataset_revision`` pin the exact corpus commit, ``source_id`` is
that corpus row's own id, and ``prompt_sha`` identifies the prompt text itself.

Example:
    PYTHONPATH=. uv run python scripts/thinking_traces/push_traces_to_hub.py \\
        --traces /results/traces_glm-5.2-fp8.jsonl \\
        --repo-id allenai/thinking-trace-lengths --config-name glm-5.2-fp8
"""

import argparse
import json
import os

import datasets
import huggingface_hub

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, help="the .jsonl written by generate_traces.py")
    parser.add_argument("--repo-id", required=True, help="target dataset repo, e.g. allenai/thinking-traces")
    parser.add_argument("--config-name", required=True, help="subset name, normally the served model name")
    parser.add_argument("--private", action="store_true", default=True)
    parser.add_argument("--public", dest="private", action="store_false")
    parser.add_argument(
        "--best-effort",
        action="store_true",
        help="log and exit 0 on failure instead of raising; used by the in-job call",
    )
    return parser.parse_args()


def push(args: argparse.Namespace) -> None:
    rows = []
    with open(args.traces) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{args.traces} has no rows")

    usable = [r for r in rows if "error" not in r]
    logger.info("%s: %d rows (%d usable)", args.traces, len(rows), len(usable))

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set; cannot push")

    huggingface_hub.create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=True, token=token)
    ds = datasets.Dataset.from_list(usable)
    ds.push_to_hub(args.repo_id, config_name=args.config_name, private=args.private, token=token)
    logger.info("pushed %d rows to %s (config %s)", len(usable), args.repo_id, args.config_name)


def main() -> None:
    args = parse_args()
    try:
        push(args)
    except Exception as exc:  # noqa: BLE001 - best-effort mode must not fail the job
        if not args.best_effort:
            raise
        logger.error("hub push failed (%s): %s", type(exc).__name__, exc)
        logger.error(
            "traces remain in %s and are still synced to the Beaker result dataset and weka; "
            "re-run this script to backfill once the problem is fixed.",
            args.traces,
        )


if __name__ == "__main__":
    main()
