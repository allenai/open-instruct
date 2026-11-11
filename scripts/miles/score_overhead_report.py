"""Compare completed score extraction runs, retaining every transport alternative.

Each column uses its own matched no-logprobs baseline; runs are never pooled.
Example: python -m scripts.miles.score_overhead_report --transport-results
runs/mc32-saturn-20261109/results --sweep-results runs/mc-sweep-saturn-20261110/results
--output-dir runs/score-overhead-comparison
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

TRANSPORT_MODES = ["none", "selected"] + [f"{mode}_{k}" for k in (32, 64, 128) for mode in ("nested", "flat", "array")]
SWEEP_MODES = ["none", "selected"] + [mode for k in (32, 64, 128) for mode in (f"array_{k}", f"mc{k}")]
CPU_FIELDS = (
    ("server_pack_ms", "Server packing", "#345995"),
    ("server_json_ms", "Server JSON", "#8ba5cf"),
    ("client_json_parse_ms", "Client JSON parse", "#b85c38"),
    ("client_array_unpack_ms", "Client arrays", "#e9ad87"),
)


def label(mode):
    if mode in ("none", "selected"):
        return {"none": "No logprobs", "selected": "Selected only"}[mode]
    if mode.startswith("mc"):
        return f"SCMC{mode[2:]}\narrays"
    transport, width = mode.split("_")
    return f"SCT{width}\n{transport}"


def load_run(path, modes, name):
    report = json.loads((path / "report.json").read_text())
    analysis = json.loads((path / report["analysis"]).read_text())
    rows = {row["mode"]: row for row in report["rows"]}
    if missing := set(modes) - set(rows):
        raise ValueError(f"Missing measured modes in {path}: {sorted(missing)}")
    return {"name": name, "source": str(path.resolve()), "modes": modes, "rows": rows, "analysis": analysis}


def render(runs, output, trainer_path=None):
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    trainer = json.loads(trainer_path.read_text()) if trainer_path else None
    fig, axes = plt.subplots(5 if trainer else 4, 2, figsize=(21, 24 if trainer else 19), layout="constrained")
    export = []
    for col, run in enumerate(runs):
        modes, rows, analysis = run["modes"], run["rows"], run["analysis"]
        xs = np.arange(len(modes))
        colors = [
            "#a54bb0"
            if m.startswith("mc")
            else "#5eaa6c"
            if m.startswith("array")
            else "#345995"
            if m.startswith("flat")
            else "#cc8044"
            if m.startswith("nested")
            else "#72808f"
            for m in modes
        ]
        values = np.array([analysis["overhead"][m]["none"]["throughput_loss_pct"] for m in modes])
        ci = np.array([analysis["overhead"][m]["none"]["throughput_loss_95pct_block_bootstrap"] for m in modes])
        ax = axes[0, col]
        ax.bar(xs, values, color=colors)
        ax.errorbar(
            xs,
            values,
            yerr=np.maximum(0, np.vstack((values - ci[:, 0], ci[:, 1] - values))),
            fmt="none",
            color="#253746",
            capsize=3,
        )
        ax.set_title(run["name"] + " · end-to-end overhead")
        ax.set_ylabel("Throughput loss vs this run's no-logprobs baseline (%)")
        ax = axes[1, col]
        bottom = np.zeros(len(modes))
        for field, description, color in CPU_FIELDS:
            value = np.array([rows[m][field] for m in modes])
            ax.bar(xs, value, bottom=bottom, label=description, color=color)
            bottom += value
        ax.set_title("HTTP packing/unpacking · absolute time and latency fraction")
        ax.set_ylabel("Mean measured CPU stage time per request (ms)")
        for x, height, mode in zip(xs, bottom, modes, strict=True):
            ax.text(
                x, height, f"{rows[mode]['serialization_fraction_pct']:.3f}%", ha="center", va="bottom", fontsize=8
            )
        ax.margins(y=0.2)
        ax.legend(fontsize=8)
        ax = axes[2, col]
        stages = [
            ("topk_extraction_stream", "SCT extraction", "#197f8d"),
            ("mc_extraction_stream", "SCMC extraction", "#a54bb0"),
            ("result_d2h_stream", "Result copy to CPU", "#e38c3e"),
        ]
        for i, (stage, description, color) in enumerate(stages):
            value = [
                analysis["sampled_gpu_stages"].get(m, {}).get(stage, {}).get("mean_cuda_span_ms", 0) for m in modes
            ]
            ax.bar(xs + (i - 1) * 0.25, value, width=0.24, color=color, label=description)
        ax.set_title("Sampled extraction/copy spans · overlapping, not additive")
        ax.set_ylabel("Mean CUDA stream span per call (ms)")
        ax.legend(fontsize=8)
        ax = axes[3, col]
        forward = [analysis["sampled_gpu_stages"].get(m, {}).get("model_forward_stream", {}) for m in modes]
        ax.bar(xs, [stage.get("mean_cuda_span_ms", 0) for stage in forward], color=colors)
        for x, stage in zip(xs, forward, strict=True):
            ax.text(
                x,
                stage.get("mean_cuda_span_ms", 0),
                f"{100 * stage.get('estimated_cuda_span_fraction_of_block_wall', 0):.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.margins(y=0.15)
        ax.set_title("Model inference · labels estimate fraction of benchmark block wall time")
        ax.set_ylabel("Mean sampled model-forward CUDA span (ms/call)")
        for ax in axes[:4, col]:
            ax.set_xticks(xs, [label(m) for m in modes], rotation=35, ha="right")
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        for mode in modes:
            row = rows[mode]
            flat = {
                "run": run["name"],
                "source": run["source"],
                "mode": mode,
                **{k: v for k, v in row.items() if k != "gpu"},
            }
            for stage, values in analysis["sampled_gpu_stages"].get(mode, {}).items():
                flat.update({f"{stage}_{k}": v for k, v in values.items()})
            export.append(flat)
    if trainer:
        labels = ["PPO/GRPO", "SCMC32 + TIS"]
        xs = np.arange(2)
        for offset, field, description in [
            (-0.25, "step_seconds", "Whole update"),
            (0, "forward_backward_seconds", "Forward/backward host span"),
            (0.25, "optimizer_seconds", "Optimizer host span"),
        ]:
            axes[4, 0].bar(
                xs + offset,
                [r["stats"][field]["mean"] for r in trainer["runs"].values()],
                width=0.25,
                label=description,
            )
        pct = trainer["overhead"]["step_seconds"]["increase_pct"]
        axes[4, 0].set_title(f"Jupiter actual training · full loss-path overhead {pct:+.2f}%")
        axes[4, 0].set_ylabel("Seconds per optimizer update (warmup excluded)")
        for offset, field, description in [
            (-0.18, "peak_allocated_bytes", "Allocated"),
            (0.18, "peak_reserved_bytes", "Reserved"),
        ]:
            axes[4, 1].bar(
                xs + offset,
                [r["stats"][field]["max"] / 2**30 for r in trainer["runs"].values()],
                width=0.36,
                label=description,
            )
        axes[4, 1].set_title("Jupiter trainer memory · 4 trainer GPUs, EP4")
        axes[4, 1].set_ylabel("Peak memory across trainer ranks (GiB)")
        for ax in axes[4]:
            ax.set_xticks(xs, labels)
            ax.legend()
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        for mode, run in trainer["runs"].items():
            export.append(
                {
                    "run": "Jupiter trainer",
                    "source": run["source"],
                    "mode": mode,
                    **{f"trainer_{key}_mean": value["mean"] for key, value in run["stats"].items()},
                }
            )
    fig.suptitle(
        "Olmo H066 · SCT and SCMC overhead, including packing alternatives\nSaturn A100 80GB · 1,024 prompts × 2 repeats · 256 generated tokens · concurrency 32",
        fontsize=16,
    )
    fig.supxlabel(
        "Nested = ordinary JSON lists; flat = flatten/base64 after list materialization; array = arrays preserved through server IPC.\nSeparate runs and paired baselines; 95% block-bootstrap intervals. HTTP percentages are fractions of request latency.\nGPU stream spans include launch gaps and overlap inference; they are not an additive decomposition of end-to-end overhead.",
        fontsize=10,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"overhead-alternatives.{suffix}", dpi=170)
    plt.close(fig)
    fields = list(dict.fromkeys(k for row in export for k in row))
    with (output / "overhead-alternatives.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(export)
    (output / "sources.json").write_text(
        json.dumps(
            [{k: run[k] for k in ("name", "source", "modes")} for run in runs]
            + (
                [
                    {
                        "name": "Jupiter trainer",
                        "source": str(trainer_path.resolve()),
                        "scope": trainer["scope"],
                        "limitations": trainer["limitations"],
                    }
                ]
                if trainer
                else []
            ),
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-results", required=True, type=Path)
    parser.add_argument("--sweep-results", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trainer-report", type=Path)
    args = parser.parse_args()
    render(
        [
            load_run(args.transport_results, TRANSPORT_MODES, "Transport comparison"),
            load_run(args.sweep_results, SWEEP_MODES, "SCT / SCMC count sweep"),
        ],
        args.output_dir,
        args.trainer_report,
    )


if __name__ == "__main__":
    main()
