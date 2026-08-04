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
# Matches both "(12.3s)" and the post-throttle "(12.3s, semaphore_wait=4.5s)".
STARTED_RE = re.compile(r"OpenSandbox sandbox started: ([0-9a-f-]+) \(([\d.]+)s(?:, semaphore_wait=([\d.]+)s)?\)")
ADOPTED_RE = re.compile(r"Adopted OpenSandbox sandbox ([0-9a-f-]+)")
CLOSING_RE = re.compile(r"Closing OpenSandbox sandbox: ([0-9a-f-]+)")
JANITOR_RE = re.compile(r"^killed ([0-9a-f-]+)")
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
                # beginning of the create call — the logged total duration minus
                # any time spent queued in the client-side start semaphore.
                create_seconds = float(started.group(2))
                if started.group(3) is not None:
                    create_seconds -= float(started.group(3))
                starts[started.group(1)] = datetime.fromtimestamp(timestamp.timestamp() - max(0.0, create_seconds))
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
                continue

            # End-of-job janitor output ("killed <id>", no timestamp): treat
            # as a close at the last seen timestamp.
            janitor = JANITOR_RE.match(line.strip())
            if janitor and janitor.group(1) not in closes and last_timestamp is not None:
                closes[janitor.group(1)] = last_timestamp

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
    unpaired_tails: list[float] = []
    for sandbox_id, started_at in starts.items():
        closed_at = closes.get(sandbox_id)
        if closed_at is not None:
            paired_seconds.append(max(0.0, (closed_at - started_at).total_seconds()))
        else:
            tail = (events["last_timestamp"] - started_at).total_seconds() if events["last_timestamp"] else 0.0
            unpaired_tails.append(min(max(0.0, tail), lifetime_cap))

    mean_life = statistics.mean(paired_seconds) if paired_seconds else statistics.mean(unpaired_tails or [0.0])

    # Ray log dedup hides both "started" and "Closing" lines, so an unpaired
    # id usually means the close was deduped, not that the sandbox leaked.
    # Reconcile globally: total starts vs total closes (both including hidden
    # multipliers) bounds how many sandboxes truly never closed. Those are
    # priced at their observed end-of-log tail (capped); every other sandbox
    # beyond the exactly-paired ones is priced at the mean paired lifetime.
    total_starts = len(starts) + events["hidden_started"]
    total_closes = len(closes) + events["hidden_closed"]
    truly_unclosed = max(0, total_starts - total_closes)
    assumed_closed = max(0, total_starts - len(paired_seconds) - truly_unclosed)
    unclosed_tail = statistics.mean(unpaired_tails) if unpaired_tails else float(lifetime_cap)

    paired_hours = sum(paired_seconds) / 3600
    assumed_hours = assumed_closed * mean_life / 3600
    unclosed_hours = truly_unclosed * min(unclosed_tail, lifetime_cap) / 3600
    total_hours = paired_hours + assumed_hours + unclosed_hours

    print(f"Profile: {cpu} vCPU + {memory_gib:.1f} GiB per sandbox -> ${hourly_rate:.4f}/sandbox-hour")
    print(f"Total sandboxes: ~{total_starts} started, ~{total_closes} closed (incl. Ray-dedup-hidden + janitor)")
    print(f"  exactly paired: {len(paired_seconds)}", end="")
    if paired_seconds:
        print(
            f" (lifetime mean={mean_life:.0f}s median={statistics.median(paired_seconds):.0f}s "
            f"max={max(paired_seconds):.0f}s)"
        )
    else:
        print()
    print(f"  closed but unpaired (dedup-hidden lines): ~{assumed_closed}, priced at mean lifetime")
    print(f"  truly unclosed (leaks): ~{truly_unclosed}, priced at end-of-log tail capped at {lifetime_cap}s")
    if events["hidden_started"] or events["hidden_closed"]:
        print("  (launch with --env RAY_DEDUP_LOGS=0 for exact per-sandbox accounting)")
    print()
    print(f"Sandbox-hours: paired={paired_hours:.2f} assumed~={assumed_hours:.2f} leaked~={unclosed_hours:.2f}")
    print(f"ESTIMATED COST: ${total_hours * hourly_rate:.2f} ({total_hours:.2f} sandbox-hours)")
    print()
    print("Not included: LB data processing, orphans invisible to the client, cluster management fee.")
    print("Cross-check against the BigQuery billing export (see docs/sandbox_management.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
