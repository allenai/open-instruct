"""Draw the figures RL_REPORT.md embeds, from a W&B history dump.

    # on a machine that can reach wandb:
    python projects/pedagogy_rm/make_figures.py --fetch --out data/armhist.json
    # anywhere:
    python projects/pedagogy_rm/make_figures.py --history data/armhist.json --figures projects/pedagogy_rm/figures

TWO CADENCES, FETCHED SEPARATELY, WHICH IS NOT AN OPTIMISATION. wandb's scan_history
returns only rows carrying every key asked for, and eval metrics are written once every
--local_eval_every steps. Asking for step metrics and eval metrics together therefore
returns the intersection - six rows out of sixty-nine - and a training curve drawn from
that would look like the run barely moved.
"""

from __future__ import annotations

import argparse
import json
import os

DIMS = ("targeted", "actionable", "elicits", "leak")
ARM_COLOUR = {"A": "#1f77b4", "B": "#d62728"}
ARM_LABEL = {"A": "arm A  (pedagogy, raw mean)", "B": "arm B  (pedagogy_z, z-scored)"}


def fetch(out: str) -> None:
    import wandb  # noqa: PLC0415

    api = wandb.Api()
    runs = {"A": ("pm48fp8k", "pedagogy"), "B": ("k77lk6tm", "pedagogy_z")}
    blob = {}
    for label, (rid, prefix) in runs.items():
        run = api.run(f"zsophia-massachusetts-institute-of-technology/pedagogy-rm/{rid}")
        have = set(run.summary.keys())
        step_keys = ["_step", "scores", "objective/kl1_avg"] + [f"scored/{prefix}/dim_{d}" for d in DIMS]
        eval_keys = ["_step", "eval/scores", "eval/sequence_lengths"]
        blob[label] = {
            "prefix": prefix,
            "step": list(run.scan_history(keys=[k for k in step_keys if k == "_step" or k in have], page_size=2000)),
            "eval": list(run.scan_history(keys=[k for k in eval_keys if k == "_step" or k in have], page_size=2000)),
        }
    with open(out, "w") as handle:
        json.dump(blob, handle)
    print(f"wrote {out}")


def series(rows: list[dict], key: str) -> tuple[list, list]:
    xs, ys = [], []
    for row in rows:
        if isinstance(row.get(key), (int, float)) and isinstance(row.get("_step"), (int, float)):
            xs.append(row["_step"])
            ys.append(row[key])
    return xs, ys


def draw(history: dict, figures: str) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    os.makedirs(figures, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

    # 1. What the probe achieves, against the control and the ceiling. The surface bar is
    # the point of the figure: a dimension whose control bar reaches its probe bar is being
    # predicted by length and punctuation, and `concise` is why one was excluded.
    probe = {
        "targeted": (0.36, 0.85, 0.94),
        "leak": (0.60, 0.84, 0.95),
        "actionable": (0.75, 0.94, 0.98),
        "elicits": (0.81, 0.95, 0.98),
        "concise\n(excluded)": (0.96, 0.97, 0.99),
    }
    fig, ax = plt.subplots(figsize=(7, 3.1))
    names = list(probe)
    xs = range(len(names))
    ax.bar([x - 0.22 for x in xs], [probe[n][0] for n in names], 0.4, label="surface features", color="#bbbbbb")
    ax.bar([x + 0.22 for x in xs], [probe[n][1] for n in names], 0.4, label="ridge on hidden states", color="#1f77b4")
    for x, n in zip(xs, names, strict=True):
        ax.plot([x - 0.45, x + 0.45], [probe[n][2]] * 2, color="#333", lw=1.2, ls="--")
    ax.plot([], [], color="#333", lw=1.2, ls="--", label="agreement ceiling")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pearson r")
    ax.set_title("Probe accuracy vs a control that knows nothing about teaching")
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "probe_accuracy.png"))
    plt.close(fig)

    # 2. Reward and length on one pair of axes, because the whole question is whether they
    # moved together. Arm B's reward is omitted rather than drawn flat at zero: a z-score
    # is mean-zero within its group by construction and the line would invite a comparison
    # that does not exist.
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True)
    xs, ys = series(history["A"]["step"], "scores")
    top.plot(xs, ys, color=ARM_COLOUR["A"], lw=1.4, label="arm A reward (train)")
    ex, ey = series(history["A"]["eval"], "eval/scores")
    top.plot(ex, ey, color=ARM_COLOUR["A"], lw=1.4, ls="--", marker="o", ms=3, label="arm A reward (held-out)")
    top.axhline(1.634, color="#888", lw=1, ls=":")
    # Headroom above the guide line so the annotation does not land on the title. The line
    # is the reference the reward is closing on, so it has to stay visible with its label.
    top.set_ylim(top=max(1.70, max(ys) + 0.06))
    top.text(1, 1.640, "socratic-style mean in the labelled corpus", fontsize=7, color="#666", va="bottom")
    top.set_ylabel("reward  [0, 2]")
    top.set_title("The reward rises and generalises; the turns get shorter")
    top.legend(loc="lower right", fontsize=8, framealpha=0.95)

    for arm in ("A", "B"):
        lx, ly = series(history[arm]["eval"], "eval/sequence_lengths")
        bottom.plot(lx, ly, color=ARM_COLOUR[arm], lw=1.4, marker="o", ms=3, label=ARM_LABEL[arm])
    bottom.set_ylabel("held-out turn length\n(tokens)")
    bottom.set_xlabel("training step")
    bottom.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "reward_and_length.png"))
    plt.close(fig)

    # 3. Per-dimension movement. Drawn as change from the first logged value so four scales
    # with different starting points can share an axis, and because the question is which
    # dimensions moved rather than where they sit.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0), sharey=True)
    for ax, arm in zip(axes, ("A", "B"), strict=True):
        prefix = history[arm]["prefix"]
        for dim in DIMS:
            xs, ys = series(history[arm]["step"], f"scored/{prefix}/dim_{dim}")
            if not ys:
                continue
            ax.plot(xs, [y - ys[0] for y in ys], lw=1.3, label=dim)
        ax.axhline(0, color="#333", lw=0.8)
        ax.set_title(ARM_LABEL[arm], fontsize=9)
        ax.set_xlabel("training step")
    axes[0].set_ylabel("change from step 1\n(rubric points, signed)")
    axes[1].legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.suptitle("Three dimensions move; targeted is pinned near its ceiling", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "dimensions.png"))
    plt.close(fig)

    # 4. Drift. The x-axis of the overoptimization result is KL from the initial policy,
    # not steps, so it is worth its own panel - and it is where the two arms differ most.
    fig, ax = plt.subplots(figsize=(7, 2.6))
    for arm in ("A", "B"):
        xs, ys = series(history[arm]["step"], "objective/kl1_avg")
        ax.plot(xs, ys, color=ARM_COLOUR[arm], lw=1.4, label=ARM_LABEL[arm])
    ax.set_xlabel("training step")
    ax.set_ylabel("KL from reference")
    ax.set_title("Arm B drifts ~3x faster at the same beta")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(figures, "kl_drift.png"))
    plt.close(fig)

    print(f"wrote 4 figures to {figures}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="pull history from wandb first")
    parser.add_argument("--history", default="data/armhist.json")
    parser.add_argument("--figures", default="projects/pedagogy_rm/figures")
    args = parser.parse_args()

    if args.fetch:
        fetch(args.history)
    with open(args.history) as handle:
        draw(json.load(handle), args.figures)


if __name__ == "__main__":
    main()
