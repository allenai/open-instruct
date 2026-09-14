"""Summarize matched retained-batch screens without hiding cold compilation."""

import argparse
import json
import statistics
from pathlib import Path


def analyze(root, warmup=6):
    root = Path(root)
    ranks = [json.loads(path.read_text()) for path in sorted(root.glob("trainer-capacity-rank*.json"))]
    if len(ranks) != 2 or {r["rank"] for r in ranks} != {0, 1}:
        raise ValueError("Expected reports from both EP2 ranks")
    ranks.sort(key=lambda r: r["rank"])
    if not all(r["passed"] for r in ranks):
        raise ValueError("Both ranks must complete; failed/partial screens are not throughput results")
    records = []
    for batch in zip(*(r["batches"] for r in ranks), strict=True):
        if len({(b["update"], b["source_sha256"]) for b in batch}) != 1:
            raise ValueError("Rank input identities differ")
        seconds = max(b["seconds"] for b in batch)
        tokens = sum(b["local_tokens"] for b in batch)
        response = sum(b["local_response_tokens"] for b in batch)
        records.append(
            dict(
                update=batch[0]["update"],
                source_sha256=batch[0]["source_sha256"],
                seconds=seconds,
                model_tokens=tokens,
                response_tokens=response,
                model_tokens_per_second_per_gpu=tokens / seconds / 2,
                response_tokens_per_second_per_gpu=response / seconds / 2,
                phases_rank0=batch[0]["phases"],
                other_seconds_rank0=batch[0]["other_seconds"],
                jit_misses_per_rank=[b["compilation"]["jit_miss_count"] for b in batch],
                cache_artifact_writes_per_rank=[b["compilation"]["cache_artifact_writes_by_extension"] for b in batch],
                unique_graphs_cumulative_per_rank=[b["dynamo_stats"].get("unique_graphs", 0) for b in batch],
                peak_allocated_bytes_per_rank=[b["memory_allocated_peak"] for b in batch],
            )
        )
    warm = [r for r in records if r["update"] >= warmup]
    if not warm:
        raise ValueError("Warm window is empty")
    elapsed = sum(r["seconds"] for r in warm)
    phases = {key for r in warm for key in r["phases_rank0"]}
    return dict(
        variant=ranks[0]["variant"],
        passed=True,
        warmup_updates=warmup,
        scope="Matched retained-batch screen; max rank wall time, summed rank tokens. Phase means use rank 0 and do not partition max-rank wall time. No inference/publication; compilation activity remains visible.",
        initialization_seconds_per_rank=[r["initialization_seconds"] for r in ranks],
        measured_updates=len(warm),
        warm_seconds=elapsed,
        warm_model_tokens_per_second_per_gpu=sum(r["model_tokens"] for r in warm) / elapsed / 2,
        warm_response_tokens_per_second_per_gpu=sum(r["response_tokens"] for r in warm) / elapsed / 2,
        mean_seconds_per_update=elapsed / len(warm),
        phase_mean_seconds_rank0={
            key: statistics.mean(r["phases_rank0"].get(key, 0) for r in warm) for key in sorted(phases)
        },
        other_mean_seconds_rank0=statistics.mean(r["other_seconds_rank0"] for r in warm),
        per_update=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--warmup", type=int, default=6)
    args = parser.parse_args()
    report = analyze(args.root, args.warmup)
    (args.root / "trainer-capacity-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "per_update"}, indent=2))


if __name__ == "__main__":
    main()
