"""Plot validated response-length summaries without reading rollout pickles."""

import argparse
import json
from pathlib import Path

from matplotlib import pyplot as plt
from scripts.miles import analyze_response_lengths as analysis


def plot(report, output):
    assert report["valid"]
    figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, constrained_layout=True)
    for backend, color in (("core", "#1565c0"), ("megatron", "#c45b12")):
        arm = report["arms"][backend]
        updates = arm["training"]
        x = [row["step"] for row in updates]
        smooth = [
            analysis.summarize(
                [sample for row in updates[max(0, index - 9) : index + 1] for sample in row["samples"]], training=True
            )
            for index in range(len(updates))
        ]
        axes[0, 0].plot(x, [row["length"]["mean"] for row in updates], color=color, alpha=0.15)
        axes[0, 0].plot(x, [row["length"]["mean"] for row in smooth], color=color, label=f"{backend} mean")
        axes[0, 0].plot(
            x, [row["length"]["median"] for row in smooth], color=color, ls="--", label=f"{backend} median"
        )
        axes[0, 1].plot(x, [row["length"]["p90"] for row in updates], color=color, alpha=0.2)
        axes[0, 1].plot(x, [row["length"]["p90"] for row in smooth], color=color, label=backend)
        axes[1, 0].plot(x, [row["mean_reward"] for row in smooth], color=color, label=f"{backend} train")
        axes[1, 1].plot(x, [row["cap_fraction"] for row in smooth], color=color, label=f"{backend} train")
        for axis, key in ((axes[1, 0], "mean_reward"), (axes[1, 1], "cap_fraction")):
            axis.plot(
                [row["step"] for row in arm["evaluation"]],
                [row[key] for row in arm["evaluation"]],
                color=color,
                marker="o",
                ls="--",
                label=f"{backend} held-out",
            )
        for key, linestyle in (("correct_length", "-"), ("wrong_length", "--")):
            axes[2, 0].plot(
                x,
                [row[key]["mean"] for row in smooth],
                color=color,
                ls=linestyle,
                label=f"{backend} {key.split('_')[0]}",
            )
        axes[2, 1].plot(
            x, [row["zero_policy_advantage_sample_fraction"] for row in smooth], color=color, label=backend
        )
    titles = (
        "Training response length: mean and median",
        "Training response length: p90 (cap 4096)",
        "Training reward and held-out accuracy",
        "Fraction of responses reaching 4096 tokens",
        "Training mean length, conditioned on reward",
        "Samples in groups with zero policy advantage (reconstructed)",
    )
    for axis, title in zip(axes.flat, titles, strict=True):
        axis.set_title(title, fontsize=11)
        axis.axvspan(50, 80, color="grey", alpha=0.10)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8, ncol=2)
        axis.set_xlim(0, 100)
    for axis in axes[2]:
        axis.set_xlabel("Completed updates at generation (training rollout index)")
    for axis in (axes[0, 0], axes[0, 1], axes[2, 0]):
        axis.set_ylabel("Tokens")
    for axis in (axes[1, 0], axes[1, 1], axes[2, 1]):
        axis.set_ylim(0, 1)
        axis.set_ylabel("Fraction")
    figure.suptitle(
        "Original matched GSM8K100: Core vs Megatron\n"
        "Training lines pool trailing 10 rollouts; faint lines are per rollout. Grey band: 50–80. One run per backend.",
        fontsize=13,
    )
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(json.loads(args.report.read_text()), args.output)


if __name__ == "__main__":
    main()
