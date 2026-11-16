"""Summarize repeated half-group measurements; bootstrap batch-level powers."""

import argparse
import json
import random
from pathlib import Path


def estimate(rows):
    signal = sum(row["signal_power"] for row in rows) / len(rows)
    noise = sum(row["full_batch_noise_power"] for row in rows) / len(rows)
    groups = {row["groups"] for row in rows}
    responses = {row["responses"] for row in rows}
    if len(groups) != 1 or len(responses) != 1:
        raise ValueError("Summarize one batch size at a time")
    return {
        "measurements": len(rows),
        "mean_signal_power": signal,
        "mean_full_batch_noise_power": noise,
        "noise_to_signal": noise / signal if signal > 0 else None,
        "noise_scale_groups": next(iter(groups)) * noise / signal if signal > 0 else None,
        "noise_scale_responses": next(iter(responses)) * noise / signal if signal > 0 else None,
    }


def summarize(rows):
    result = estimate(rows)
    rng = random.Random(17)
    draws = [estimate(rng.choices(rows, k=len(rows))) for _ in range(4000)]
    scales = sorted(d["noise_scale_responses"] for d in draws if d["noise_scale_responses"] is not None)
    result["bootstrap_positive_signal_fraction"] = len(scales) / len(draws)
    result["batch_bootstrap95_noise_scale_responses"] = (
        [scales[100], scales[3899]] if len(scales) == len(draws) else None
    )
    result["mean_extra_backward_and_snapshot_seconds"] = sum(
        row["half_backward_and_snapshot_seconds"] for row in rows
    ) / len(rows)
    result["mean_full_backward_seconds"] = sum(row["full_backward_seconds"] for row in rows) / len(rows)
    result["mean_statistics_seconds"] = sum(row["statistics_seconds"] for row in rows) / len(rows)
    result["limitations"] = (
        "Eight-batch fixed-weight local diagnostic on accepted priority-sampled groups; "
        "bootstrap resamples batches and does not establish an optimal Adam batch size. "
        "Do not average per-batch noise/signal ratios or pool different checkpoints."
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    options = parser.parse_args()
    rows = [json.loads(line) for line in options.measurements.read_text().splitlines() if line]
    if not rows:
        raise ValueError("No gradient measurements")
    print(json.dumps(summarize(rows), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
