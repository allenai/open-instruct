#!/usr/bin/env python3
"""Estimate the GKE Autopilot cost of OpenSandbox sandboxes from a training log.

Pairs the ``OpenSandbox sandbox started: <id>`` and ``Closing OpenSandbox
sandbox: <id>`` lines that OpenSandboxBackend logs into per-sandbox lifetimes,
then prices total sandbox-hours at Autopilot per-request rates. Because
Autopilot bills pod resource *requests* per second, this client-side estimate
tracks the real bill closely; comparing it against the BigQuery billing export
shows how much goes to things the client can't see (orphans, scheduling gaps,
LB data processing). See docs/sandbox_management.md.

Usage:
    python scripts/opensandbox/estimate_sandbox_cost.py <beaker-job-log> \
        [--vcpu-rate 0.0445] [--gib-rate 0.0049] [--cpu N] [--memory-gib N]

cpu / memory / lifetime-cap default to the values parsed from the backend's
own "Starting OpenSandbox sandbox (...)" log lines; rates default to GKE
Autopilot general-purpose on-demand list prices (us-central1, 2026-07).

Caveats printed with the result:
- Ray deduplicates similar log lines ("[repeated Nx across cluster]"), which
  hides sandbox ids; hidden events are counted from the multipliers and priced
  at the mean observed lifetime. Launch with --env RAY_DEDUP_LOGS=0 for exact
  per-sandbox accounting.
- Sandboxes with no Closing line (job killed, log truncated) are priced up to
  the last log timestamp, capped at the sandbox lifetime.
"""

import argparse
import re
import statistics
import sys
from datetime import datetime

TIMESTAMP_RE = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
STARTING_RE = re.compile(r"Starting OpenSandbox sandbox \(.*?cpu=([\d.]+), memory_mib=(\d+), lifetime=(\d+)s\)")
STARTED_RE = re.compile(r"OpenSandbox sandbox started: ([0-9a-f-]+) \(([\d.]+)s\)")
ADOPTED_RE = re.compile(r"Adopted OpenSandbox sandbox ([0-9a-f-]+)")
CLOSING_RE = re.compile(r"Closing OpenSandbox sandbox: ([0-9a-f-]+)")
REPEATED_RE = re.compile(r"\[repeated (\d+)x across cluster\]")


def parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if match is None:
        return None
    return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")


def parse_log(path: str) -> dict:
    """Scan the log once and collect sandbox lifecycle events."""
    starts: dict[str, datetime] = {}  # sandbox id -> billing start (create call start)
    closes: dict[str, datetime] = {}
    hidden_started = 0  # events hidden by Ray log dedup
    hidden_closed = 0
    detected: dict[str, float] = {}
    last_timestamp: datetime | None = None

    with open(path, errors="replace") as f:
        for line in f:
            timestamp = parse_timestamp(line)
            if timestamp is not None:
                last_timestamp = timestamp

            if not detected:
                config = STARTING_RE.search(line)
                if config:
                    detected = {
                        "cpu": float(config.group(1)),
                        "memory_gib": int(config.group(2)) / 1024,
                        "lifetime_s": int(config.group(3)),
                    }

            repeated = REPEATED_RE.search(line)
            multiplier = int(repeated.group(1)) if repeated else 1

            started = STARTED_RE.search(line)
            if started and timestamp is not None:
                # Billing starts roughly when the pod is scheduled, i.e. at the
                # beginning of the create call — the logged duration before the
                # "started" line.
                create_seconds = float(started.group(2))
                starts[started.group(1)] = datetime.fromtimestamp(timestamp.timestamp() - create_seconds)
                hidden_started += multiplier - 1
                continue

            adopted = ADOPTED_RE.search(line)
            if adopted and timestamp is not None:
                starts[adopted.group(1)] = timestamp
                hidden_started += multiplier - 1
                continue

            closing = CLOSING_RE.search(line)
            if closing and timestamp is not None:
                closes[closing.group(1)] = timestamp
                hidden_closed += multiplier - 1

    return {
        "starts": starts,
        "closes": closes,
        "hidden_started": hidden_started,
        "hidden_closed": hidden_closed,
        "detected": detected,
        "last_timestamp": last_timestamp,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="Path to the training job log")
    parser.add_argument("--cpu", type=float, default=None, help="vCPU per sandbox (default: parsed from log)")
    parser.add_argument("--memory-gib", type=float, default=None, help="GiB per sandbox (default: parsed from log)")
    parser.add_argument(
        "--lifetime-cap", type=int, default=None, help="Sandbox lifetime cap in seconds (default: parsed from log)"
    )
    parser.add_argument("--vcpu-rate", type=float, default=0.0445, help="$/vCPU-hour (Autopilot on-demand)")
    parser.add_argument("--gib-rate", type=float, default=0.0049, help="$/GiB-hour (Autopilot on-demand)")
    args = parser.parse_args()

    events = parse_log(args.log)
    starts, closes = events["starts"], events["closes"]
    detected = events["detected"]

    cpu = args.cpu if args.cpu is not None else detected.get("cpu", 1.0)
    memory_gib = args.memory_gib if args.memory_gib is not None else detected.get("memory_gib", 4.0)
    lifetime_cap = args.lifetime_cap if args.lifetime_cap is not None else int(detected.get("lifetime_s", 3600))
    hourly_rate = cpu * args.vcpu_rate + memory_gib * args.gib_rate

    if not starts:
        print("No OpenSandbox sandbox lifecycle events found in the log.")
        return 1

    paired_seconds: list[float] = []
    unclosed_seconds: list[float] = []
    for sandbox_id, started_at in starts.items():
        closed_at = closes.get(sandbox_id)
        if closed_at is not None:
            paired_seconds.append(max(0.0, (closed_at - started_at).total_seconds()))
        else:
            # No Closing line: price up to end of log, bounded by the cap.
            tail = (events["last_timestamp"] - started_at).total_seconds() if events["last_timestamp"] else 0.0
            unclosed_seconds.append(min(max(0.0, tail), lifetime_cap))

    mean_life = statistics.mean(paired_seconds) if paired_seconds else statistics.mean(unclosed_seconds or [0.0])
    hidden_count = max(events["hidden_started"], events["hidden_closed"])
    hidden_seconds = hidden_count * mean_life

    observed_hours = sum(paired_seconds) / 3600
    unclosed_hours = sum(unclosed_seconds) / 3600
    hidden_hours = hidden_seconds / 3600
    total_hours = observed_hours + unclosed_hours + hidden_hours

    print(f"Profile: {cpu} vCPU + {memory_gib:.1f} GiB per sandbox -> ${hourly_rate:.4f}/sandbox-hour")
    print(f"Observed sandboxes (started+closed): {len(paired_seconds)}")
    if paired_seconds:
        print(
            f"  lifetime mean={mean_life:.0f}s median={statistics.median(paired_seconds):.0f}s "
            f"max={max(paired_seconds):.0f}s"
        )
    print(f"Unclosed sandboxes (no Closing line, capped at {lifetime_cap}s): {len(unclosed_seconds)}")
    if hidden_count:
        print(f"Events hidden by Ray log dedup: ~{hidden_count} (priced at mean lifetime; ")
        print("  relaunch with --env RAY_DEDUP_LOGS=0 for exact accounting)")
    print()
    print(f"Sandbox-hours: observed={observed_hours:.2f} unclosed={unclosed_hours:.2f} hidden~={hidden_hours:.2f}")
    print(f"ESTIMATED COST: ${total_hours * hourly_rate:.2f} ({total_hours:.2f} sandbox-hours)")
    print()
    print("Not included: LB data processing, orphans invisible to the client, cluster management fee.")
    print("Cross-check against the BigQuery billing export (see docs/sandbox_management.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
