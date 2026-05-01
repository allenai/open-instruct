"""Compare metrics between two wandb runs.

Auto-discovers the intersection of logged metric keys and reports per-metric
summary statistics (n, mean, median) plus the new/old ratio. Fetches each key
independently so metrics logged on different cadences are not dropped.
"""

import argparse
import logging
import math

import numpy as np
import wandb

logger = logging.getLogger(__name__)


def discover_keys(run: wandb.apis.public.Run) -> set[str]:
    keys: set[str] = set()
    for k, v in run.summary.items():
        if k.startswith("_"):
            continue
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            keys.add(k)
    return keys


def fetch_keys(run: wandb.apis.public.Run, keys: list[str], max_steps: int | None) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {k: [] for k in keys}
    for row in run.scan_history(keys=[*keys, "_step"], page_size=2000):
        step = row.get("_step")
        if max_steps is not None and step is not None and step > max_steps:
            continue
        for key in keys:
            v = row.get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                continue
            f = float(v)
            if math.isnan(f):
                continue
            values[key].append(f)
    return values


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-run", required=True, help="entity/project/run_id")
    parser.add_argument("--old-run", required=True, help="entity/project/run_id")
    parser.add_argument("--max-steps", type=int, default=None, help="Cap _step <= this for both runs")
    parser.add_argument("--filter", default=None, help="Only include keys containing this substring")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    api = wandb.Api()
    new_run = api.run(args.new_run)
    old_run = api.run(args.old_run)

    new_keys = discover_keys(new_run)
    old_keys = discover_keys(old_run)
    if args.filter:
        new_keys = {k for k in new_keys if args.filter in k}
        old_keys = {k for k in old_keys if args.filter in k}

    only_new = sorted(new_keys - old_keys)
    only_old = sorted(old_keys - new_keys)
    shared = sorted(new_keys & old_keys)

    logger.info("Keys only in NEW (%d):", len(only_new))
    for k in only_new:
        logger.info("  + %s", k)
    logger.info("")
    logger.info("Keys only in OLD (%d):", len(only_old))
    for k in only_old:
        logger.info("  - %s", k)
    logger.info("")
    logger.info("Shared keys (%d):", len(shared))
    logger.info("")
    logger.info(
        "%-44s %5s %10s %10s   %5s %10s %10s   %s",
        "metric",
        "n_new",
        "mean_new",
        "med_new",
        "n_old",
        "mean_old",
        "med_old",
        "ratio",
    )
    logger.info("-" * 120)

    new_values = fetch_keys(new_run, shared, args.max_steps)
    old_values = fetch_keys(old_run, shared, args.max_steps)

    for key in shared:
        new_summary = summarize(new_values[key])
        old_summary = summarize(old_values[key])
        if new_summary["n"] == 0 or old_summary["n"] == 0:
            logger.info(
                "%-44s n_new=%d n_old=%d (no data in window)", key, new_summary["n"], old_summary["n"]
            )
            continue
        ratio = new_summary["mean"] / old_summary["mean"] if old_summary["mean"] != 0 else float("inf")
        logger.info(
            "%-44s %5d %10.4g %10.4g   %5d %10.4g %10.4g   %.3fx",
            key,
            new_summary["n"],
            new_summary["mean"],
            new_summary["median"],
            old_summary["n"],
            old_summary["mean"],
            old_summary["median"],
            ratio,
        )


if __name__ == "__main__":
    main()
