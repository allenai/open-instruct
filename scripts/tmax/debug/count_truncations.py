#!/usr/bin/env python3
"""Parse rollouts dumped by ``--save_traces`` and report truncation stats.

Companion to ``scripts/tmax/debug/count_truncations.sh``: that launcher does
one full grpo_fast.py step (100 prompts x 8 rollouts) with ``--save_traces``,
which writes JSONL shards via :func:`open_instruct.rl_utils._save_rollouts`.
This script reads those shards and prints how many rollouts hit the response
length cap, broken down by ``finish_reason``, with optional per-prompt
attribution.

Usage:
    python scripts/tmax/debug/count_truncations.py /path/to/rollouts_dir \
        --response-length 32768 \
        [--run-name swerl_qwen35_9b_count_truncations] \
        [--per-prompt]

The ``rollouts_dir`` should be the directory passed to ``--rollouts_save_path``.
Files are matched as ``*_rollouts_*.jsonl`` (filtered by ``--run-name`` if given).
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics
import sys


def iter_rollout_records(rollouts_dir: str, run_name: str | None) -> collections.abc.Iterator[dict]:
    pattern = os.path.join(rollouts_dir, "*_rollouts_*.jsonl")
    files = sorted(glob.glob(pattern))
    if run_name:
        files = [f for f in files if os.path.basename(f).startswith(f"{run_name}_rollouts_")]
    if not files:
        raise SystemExit(f"No rollout shards matched {pattern}" + (f" run_name={run_name}" if run_name else ""))
    print(f"Reading {len(files)} shard(s):", file=sys.stderr)
    for f in files:
        print(f"  {f}", file=sys.stderr)
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def percentile(sorted_vals: list[int], pct: float) -> int:
    if not sorted_vals:
        return 0
    idx = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("rollouts_dir", help="Directory containing *_rollouts_*.jsonl files")
    p.add_argument(
        "--response-length",
        type=int,
        default=32768,
        help="Hard cap used at generation time. Rollouts >= this many response tokens are 'at-cap' (default: 32768).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional run-name prefix to filter shards by (e.g. swerl_qwen35_9b_count_truncations__42__1714000000)",
    )
    p.add_argument(
        "--per-prompt",
        action="store_true",
        help="Also print prompts whose entire group of rollouts was truncated.",
    )
    args = p.parse_args()

    finish_counter: collections.Counter[str] = collections.Counter()
    response_lens: list[int] = []
    at_cap_total = 0
    by_prompt_total: dict[int, int] = collections.defaultdict(int)
    by_prompt_at_cap: dict[int, int] = collections.defaultdict(int)
    rewards_at_cap: list[float] = []
    rewards_not_at_cap: list[float] = []

    for rec in iter_rollout_records(args.rollouts_dir, args.run_name):
        finish_counter[rec.get("finish_reason", "<missing>")] += 1
        n_resp_tokens = len(rec.get("response_tokens", []) or [])
        response_lens.append(n_resp_tokens)
        prompt_idx = int(rec.get("prompt_idx", -1))
        by_prompt_total[prompt_idx] += 1
        is_at_cap = n_resp_tokens >= args.response_length
        if is_at_cap:
            at_cap_total += 1
            by_prompt_at_cap[prompt_idx] += 1
            rewards_at_cap.append(float(rec.get("reward", 0.0)))
        else:
            rewards_not_at_cap.append(float(rec.get("reward", 0.0)))

    total = len(response_lens)
    if total == 0:
        raise SystemExit("Found 0 rollout records.")

    response_lens.sort()
    print("=" * 64)
    print(f"Total rollouts: {total}")
    print(f"Unique prompts: {len(by_prompt_total)}")
    print()
    print(f"At length cap (>= {args.response_length} response tokens): "
          f"{at_cap_total} ({100.0 * at_cap_total / total:.2f}%)")
    print()
    print("Finish reason breakdown:")
    for reason, count in sorted(finish_counter.items(), key=lambda kv: -kv[1]):
        print(f"  {reason:>16s}: {count:5d}  ({100.0 * count / total:.2f}%)")
    print()
    print("Response length stats (tokens):")
    print(f"  min  : {response_lens[0]}")
    print(f"  p25  : {percentile(response_lens, 25)}")
    print(f"  p50  : {percentile(response_lens, 50)}")
    print(f"  p75  : {percentile(response_lens, 75)}")
    print(f"  p90  : {percentile(response_lens, 90)}")
    print(f"  p95  : {percentile(response_lens, 95)}")
    print(f"  p99  : {percentile(response_lens, 99)}")
    print(f"  max  : {response_lens[-1]}")
    print(f"  mean : {statistics.mean(response_lens):.1f}")
    print()
    if rewards_at_cap or rewards_not_at_cap:
        print("Reward summary (mean):")
        if rewards_at_cap:
            n_correct = sum(1 for r in rewards_at_cap if r > 0)
            print(f"  at-cap     : n={len(rewards_at_cap):4d}  "
                  f"mean={statistics.mean(rewards_at_cap):.3f}  "
                  f"correct(>0)={n_correct} ({100.0 * n_correct / len(rewards_at_cap):.2f}%)")
        if rewards_not_at_cap:
            n_correct = sum(1 for r in rewards_not_at_cap if r > 0)
            print(f"  not-at-cap : n={len(rewards_not_at_cap):4d}  "
                  f"mean={statistics.mean(rewards_not_at_cap):.3f}  "
                  f"correct(>0)={n_correct} ({100.0 * n_correct / len(rewards_not_at_cap):.2f}%)")
    print()

    fully_truncated = [pi for pi, total_ in by_prompt_total.items() if by_prompt_at_cap.get(pi, 0) == total_]
    print(f"Prompts where ALL rollouts hit the cap: {len(fully_truncated)} / {len(by_prompt_total)}")
    print(f"Prompts where SOME rollout hit the cap: "
          f"{sum(1 for pi in by_prompt_total if by_prompt_at_cap.get(pi, 0) > 0)} / {len(by_prompt_total)}")

    if args.per_prompt:
        print()
        print("Per-prompt truncation rates (at-cap / total):")
        for pi in sorted(by_prompt_total.keys()):
            tot = by_prompt_total[pi]
            cap = by_prompt_at_cap.get(pi, 0)
            bar = "#" * cap + "." * (tot - cap)
            print(f"  prompt_idx={pi:4d}  {cap:2d}/{tot:2d}  [{bar}]")


if __name__ == "__main__":
    main()
