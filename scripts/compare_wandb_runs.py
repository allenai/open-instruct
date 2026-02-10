#!/usr/bin/env python
"""Compare metrics between two wandb runs (new vs old format)."""

import argparse
import logging

import numpy as np
import wandb


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CANONICAL_METRICS = {
    "train/token_count": {
        "new": "train/token_count",
        "old": None,
    },
    "train/rewards_chosen": {
        "new": "train/rewards_chosen",
        "old": "rewards/chosen",
    },
    "train/rewards_rejected": {
        "new": "train/rewards_rejected",
        "old": "rewards/rejected",
    },
    "train/rewards_margin": {
        "new": "train/rewards_margin",
        "old": "rewards/margin",
    },
    "train/rewards_accuracy": {
        "new": "train/rewards_accuracy",
        "old": "rewards/accuracy",
    },
    "train/logps_chosen": {
        "new": "train/logps_chosen",
        "old": "logps/chosen",
    },
    "train/logps_rejected": {
        "new": "train/logps_rejected",
        "old": "logps/rejected",
    },
    "train/loss": {
        "new": "train/loss",
        "old": "train_loss",
    },
    "optim/LR": {
        "new": "optim/LR",
        "old": "learning_rate",
    },
}


def fetch_run_history(run_path: str) -> dict[str, list[tuple[int, float]]]:
    """Fetch history for a wandb run, returning {metric_name: [(step, value), ...]}."""
    api = wandb.Api()
    run = api.run(run_path)

    history = {}
    for row in run.scan_history():
        step = row.get("_step", 0)
        for key, value in row.items():
            if key.startswith("_"):
                continue
            if value is None:
                continue
            if key not in history:
                history[key] = []
            history[key].append((step, value))

    return history


def compare_runs(new_run_path: str, old_run_path: str, rtol: float = 1e-5, atol: float = 1e-8) -> dict:
    """Compare metrics between new and old runs."""
    logger.info(f"Fetching new run: {new_run_path}")
    new_history = fetch_run_history(new_run_path)

    logger.info(f"Fetching old run: {old_run_path}")
    old_history = fetch_run_history(old_run_path)

    results = {}

    for canonical_name, mapping in CANONICAL_METRICS.items():
        new_key = mapping["new"]
        old_key = mapping["old"]

        if old_key is None:
            logger.info(f"Skipping {canonical_name}: no old metric mapping")
            results[canonical_name] = {"status": "skipped", "reason": "no old metric"}
            continue

        if new_key not in new_history:
            logger.warning(f"Missing {new_key} in new run")
            results[canonical_name] = {"status": "missing", "reason": f"{new_key} not in new run"}
            continue

        if old_key not in old_history:
            logger.warning(f"Missing {old_key} in old run")
            results[canonical_name] = {"status": "missing", "reason": f"{old_key} not in old run"}
            continue

        new_data = sorted(new_history[new_key], key=lambda x: x[0])
        old_data = sorted(old_history[old_key], key=lambda x: x[0])

        new_steps = [s for s, _ in new_data]
        old_steps = [s for s, _ in old_data]
        new_values = np.array([v for _, v in new_data])
        old_values = np.array([v for _, v in old_data])

        if len(new_values) != len(old_values):
            logger.warning(
                f"{canonical_name}: step count mismatch - new={len(new_values)}, old={len(old_values)}, "
                f"comparing first {min(len(new_values), len(old_values))} steps"
            )
            min_len = min(len(new_values), len(old_values))
            new_values = new_values[:min_len]
            old_values = old_values[:min_len]
            new_steps = new_steps[:min_len]
            old_steps = old_steps[:min_len]

        if new_steps != old_steps:
            logger.warning(f"{canonical_name}: steps don't align")
            results[canonical_name] = {"status": "mismatch", "reason": "steps don't align"}
            continue

        diffs = np.abs(new_values - old_values)
        rel_diffs = diffs / (np.abs(old_values) + 1e-10)
        close = np.allclose(new_values, old_values, rtol=rtol, atol=atol)
        max_diff = np.max(diffs)
        max_rel_diff = np.max(rel_diffs)

        results[canonical_name] = {
            "status": "match" if close else "differ",
            "num_steps": len(new_values),
            "max_abs_diff": float(max_diff),
            "max_rel_diff": float(max_rel_diff),
            "new_mean": float(np.mean(new_values)),
            "old_mean": float(np.mean(old_values)),
        }

        status = "MATCH" if close else "DIFFER"
        logger.info(
            f"{canonical_name}: {status} "
            f"(steps={len(new_values)}, max_abs_diff={max_diff:.2e}, max_rel_diff={max_rel_diff:.2e})"
        )

        logger.info(f"  {'Step':>5s}  {'New':>12s}  {'Old':>12s}  {'AbsDiff':>12s}  {'RelDiff':>12s}")
        for i in range(len(new_values)):
            logger.info(
                f"  {new_steps[i]:5d}  {new_values[i]:12.6f}  {old_values[i]:12.6f}  "
                f"{diffs[i]:12.2e}  {rel_diffs[i]:12.2e}"
            )

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare metrics between two wandb runs")
    parser.add_argument("--new-run", required=True, help="New run path (e.g., entity/project/run_id)")
    parser.add_argument("--old-run", required=True, help="Old run path (e.g., entity/project/run_id)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for comparison")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for comparison")
    args = parser.parse_args()

    results = compare_runs(args.new_run, args.old_run, rtol=args.rtol, atol=args.atol)

    logger.info("")
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    match_count = sum(1 for r in results.values() if r.get("status") == "match")
    differ_count = sum(1 for r in results.values() if r.get("status") == "differ")
    skip_count = sum(1 for r in results.values() if r.get("status") in ("skipped", "missing", "mismatch"))

    logger.info(f"Matched: {match_count}")
    logger.info(f"Differed: {differ_count}")
    logger.info(f"Skipped/Missing: {skip_count}")

    if differ_count > 0:
        logger.info("")
        logger.info("Metrics that differ:")
        for name, r in results.items():
            if r.get("status") == "differ":
                logger.info(f"  - {name}: max_abs_diff={r['max_abs_diff']:.2e}, max_rel_diff={r['max_rel_diff']:.2e}")


if __name__ == "__main__":
    main()
