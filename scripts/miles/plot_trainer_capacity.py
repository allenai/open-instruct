"""Render the matched trainer screen; keep cold and late compilation visible."""

import argparse
import json
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg")
LABELS = {
    "baseline": "Diagnostic baseline",
    "lean": "Lean",
    "no-recompute": "Lean + no recomputation",
    "optimizer-compile": "Lean + compiled optimizer",
    "reduce-scatter": "Lean + reduce-scatter",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text())
    names = list(LABELS)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    colors = {"scoring": "#d9a441", "forward_loss_backward": "#397f9d", "optimizer": "#7c6ba6", "other": "#9badae"}
    offsets = [0.0] * len(names)
    for phase, color in colors.items():
        values = [
            data[n]["other_mean_seconds_rank0"]
            if phase == "other"
            else data[n]["phase_mean_seconds_rank0"].get(phase, 0)
            for n in names
        ]
        axes[0, 0].barh([LABELS[n] for n in names], values, left=offsets, label=phase.replace("_", " "), color=color)
        offsets = [a + b for a, b in zip(offsets, values, strict=True)]
    axes[0, 0].invert_yaxis()
    axes[0, 0].set_xlabel("Mean rank-zero seconds / update")
    axes[0, 0].set_title("Updates 6–15, including late compilation")
    axes[0, 0].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    axes[0, 1].barh(
        [LABELS[n] for n in names], [data[n]["warm_model_tokens_per_second_per_gpu"] for n in names], color="#397f9d"
    )
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Model tokens / second / GPU")
    axes[0, 1].set_title("Total tokens ÷ slower-rank elapsed time ÷ 2 GPUs")
    for name in names:
        rows = data[name]["per_update"]
        axes[1, 0].plot([r["update"] for r in rows], [r["seconds"] for r in rows], marker=".", label=LABELS[name])
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("Update index")
    axes[1, 0].set_ylabel("Trainer seconds (log scale)")
    axes[1, 0].set_title("Cold first call and later shape-bucket discovery")
    axes[1, 0].legend(fontsize=8)
    rows = data["baseline"]["per_update"]
    for rank in [0, 1]:
        axes[1, 1].plot(
            [r["update"] for r in rows],
            [r["cache_artifact_writes_per_rank"][rank].get(".cubin", 0) for r in rows],
            marker="o",
            label=f"Rank {rank}",
        )
    axes[1, 1].set_yscale("symlog", linthresh=1)
    axes[1, 1].set_xlabel("Update index")
    axes[1, 1].set_ylabel("Observed new Triton cubins (symlog scale)")
    axes[1, 1].set_title("Baseline: new convolution buckets at 7, 10 and 15")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.18)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Matched EP2 trainer screen · same retained batches · no inference or weight delivery\nModel compilation excluded: failed scoring-skip correctness gate",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
