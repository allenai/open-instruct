"""Publish derived capacity runs and a W&B report without altering training runs.

uv run --with wandb-workspaces python -m scripts.miles.publish_capacity_report \
  --run NAME=/downloaded/run --output /tmp/capacity-report --publish
Without --publish, prepare JSON only. SDK dependencies are reporting-only.
"""

import argparse
import hashlib
import json
from pathlib import Path

import wandb
from scripts.miles import capacity_metrics
from wandb_workspaces.reports import v2 as wr

SECTIONS = {
    "Trainer": {
        "Forward/backward/optimizer model tokens/sec/GPU": ["trainer/model_tokens_per_gpu_second"],
        "Scoring model tokens/sec/GPU": ["trainer/scoring_model_tokens_per_gpu_second"],
        "Trainer phase seconds": [
            "trainer/scoring_seconds",
            "trainer/forward_backward_optimizer_seconds",
            "trainer/score_train_seconds",
        ],
        "Trainer kernel activity — not SM/FLOP utilization": ["trainer/kernel_activity_percent_mean"],
        "Trainer memory GiB": ["trainer/memory_used_gib_mean", "trainer/memory_used_gib_peak_any_gpu"],
        "Useful response tokens/sec/trainer GPU over full cycle": [
            "trainer/useful_response_tokens_per_gpu_cycle_second"
        ],
    },
    "Inference": {
        "Observed decode gauge tokens/sec/GPU": ["inference/observed_decode_tokens_per_gpu_second"],
        "Useful consumed response tokens/sec/inference GPU": ["inference/useful_response_tokens_per_gpu_cycle_second"],
        "Running sequences per engine": [
            "inference/num_running_reqs_mean_per_engine",
            "inference/num_running_reqs_peak_any_engine",
        ],
        "Engine waiting requests": ["inference/num_queue_reqs_mean_per_engine"],
        "Inference kernel activity — not SM/FLOP utilization": ["inference/kernel_activity_percent_mean"],
        "Inference memory GiB": ["inference/memory_used_gib_mean", "inference/memory_used_gib_peak_any_gpu"],
        "Pool occupancy peak fraction": [
            "inference/full_token_usage_peak_any_engine",
            "inference/mamba_usage_peak_any_engine",
        ],
        "Prefix cache hit-rate gauge": ["inference/cache_hit_rate_mean_per_engine"],
    },
    "Pipeline": {
        "Useful response tokens/sec": ["pipeline/useful_response_tokens_per_second"],
        "Useful response tokens/sec/allocated GPU": ["pipeline/useful_response_tokens_per_allocated_gpu_second"],
        "Cycle components, seconds": [
            "pipeline/completed_queue_get_seconds",
            "pipeline/other_collection_seconds",
            "trainer/score_train_seconds",
            "pipeline/publication_seconds",
        ],
        "Owned versus completed groups": [
            "pipeline/producer_owned_groups",
            "pipeline/completed_queue_groups",
            "pipeline/completed_queue_capacity_groups",
        ],
        "HTTP outstanding, waiting and capacity": [
            "pipeline/http_active_requests",
            "pipeline/http_waiting_requests",
            "pipeline/http_capacity_requests",
        ],
        "Driver scoring/training/publication fraction": ["pipeline/trainer_occupied_fraction"],
    },
    "Drops and freshness": {
        "Discarded tokens versus response attempts": [
            "quality/dropped_response_tokens_fraction",
            "quality/dropped_samples_fraction",
        ],
        "Dropped attempts by length": [
            f"quality/dropped_samples_fraction_by_length/{b}"
            for b in ("0_255", "256_511", "512_1023", "1024_2047", "2048_4095", "4096_8191")
        ],
        "Dropped attempts by age": [
            f"quality/dropped_samples_by_age/{b}" for b in ("0", "1", "2", "3", "4", "5_8", "9_16", "17_plus")
        ],
        "Fraction of tokens sampled under current trainer policy": ["quality/current_policy_token_fraction"],
        "Mixed-policy responses": ["quality/mixed_responses"],
    },
    "Warmup and observation coverage": {
        "Cold and warm trainer phase seconds": [
            "trainer/scoring_seconds",
            "trainer/forward_backward_optimizer_seconds",
        ],
        "GPU minimum observation coverage": [
            f"{role}/kernel_activity_percent_minimum_coverage" for role in ("trainer", "inference")
        ],
        "Queue observation coverage": [
            "pipeline/completed_queue_groups_coverage",
            "pipeline/http_active_requests_coverage",
        ],
    },
}


def report_blocks(entity, project, group, warmup):
    blocks = [
        wr.P(
            text="Postprocessed completed-run measurements, not live telemetry. Update is 1-based. Main panels exclude the first "
            + str(warmup)
            + " updates; warmup panels retain them. Trainer phase rates count global model-input tokens once. Useful rates count consumed response tokens over the full awaited driver cycle. Observed decode gauges include work not necessarily consumed. Kernel activity is not warp occupancy or achieved FLOPs. Missing coverage is not zero activity."
        )
    ]
    for title, panels in SECTIONS.items():
        blocks.append(wr.H1(text=title))
        plots = [
            wr.LinePlot(
                title=label,
                layout=wr.Layout(x=(index % 2) * 12, y=(index // 2) * 8, w=12, h=8),
                x="update",
                y=keys,
                range_x=(None, None) if title.startswith("Warmup") else (warmup + 1, None),
                smoothing_factor=0,
                max_runs_to_show=20,
            )
            for index, (label, keys) in enumerate(panels.items())
        ]
        blocks.append(
            wr.PanelGrid(
                runsets=[wr.Runset(entity=entity, project=project, filters=f'group == "{group}"')], panels=plots
            )
        )
    return blocks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="NAME=/downloaded/run")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--entity", default="ai2-llm")
    parser.add_argument("--project", default="olmo-rl-comparison")
    parser.add_argument("--group", default="throughput-capacity-20260913")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--report-url", help="Update this capacity report after adding a completed trial")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for item in args.run:
        name, root = item.split("=", 1)
        if not name or Path(name).name != name:
            parser.error("run name must be a nonempty filename component")
        result = capacity_metrics.measurements(root, warmup=args.warmup)
        path = args.output / (name + ".json")
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        prepared.append((name, root, path, result))
    blocks = report_blocks(args.entity, args.project, args.group, args.warmup)
    if not args.publish:
        print(json.dumps({"prepared": [str(p) for _, _, p, _ in prepared], "sections": list(SECTIONS)}))
        return
    receipt = {"runs": []}
    for name, root, path, result in prepared:
        identity = hashlib.sha256((args.group + name + str(Path(root).resolve())).encode()).hexdigest()[:16]
        with wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.group,
            job_type="capacity-analysis",
            id=identity,
            resume="never",
            mode="online",
            name=name,
            dir=str(args.output),
            config={
                "source_run_root": root,
                "source_training_run_ids": sorted(
                    {p.name.rsplit("-", 1)[-1] for p in (Path(root) / "wandb").rglob("run-*") if p.is_dir()}
                ),
                "postprocessed": True,
                "warmup_updates": args.warmup,
                "allocation": result["allocation"],
            },
        ) as run:
            run.define_metric("update")
            for prefix in ("trainer", "inference", "pipeline", "quality"):
                run.define_metric(prefix + "/*", step_metric="update")
            for row in result["rows"]:
                run.log(row)
            run.save(str(path), base_path=str(args.output), policy="now")
            receipt["runs"].append({"name": name, "url": run.url, "id": identity})
    report = (
        wr.Report.from_url(args.report_url)
        if args.report_url
        else wr.Report(
            entity=args.entity,
            project=args.project,
            title="MILES / OLMo-core capacity: trainer, inference and queues",
            description="Measured throughput, capacity, memory, waste and freshness for the September 13 topology trials.",
        )
    )
    report.blocks = blocks
    report.save()
    receipt["report_url"] = report.url
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
