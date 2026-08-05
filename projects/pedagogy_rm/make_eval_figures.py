"""Draw the blinded-evaluation figures RL_REPORT.md embeds.

    python projects/pedagogy_rm/make_eval_figures.py

SEPARATE FROM make_figures.py, WHICH IS A DEPENDENCY BOUNDARY RATHER THAN A PREFERENCE.
That one reads a W&B history dump and needs the network to refresh; this one reads the
label files, the key and the pool, all of which are on disk. Keeping them apart means the
eval figures can be regenerated on a laptop with no credentials, which is where they are
usually being looked at.

WHAT THESE FOUR ARE FOR. The report's numbers are all differences between arms, and a
difference is the one thing a table communicates badly: the reader has to hold six rows in
their head and mentally attach an interval to each. Each figure here is one claim.

  1. gain_three_ways   - did the raters pay what the probe charged? the anti-hacking check
  2. length_split      - the regression, which a mean cannot show because 2 is the ideal
  3. dimension_gaps    - which dimensions moved, with intervals, both panels at once
  4. agent_trust       - why the agent labels were not used to refit
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import os
import statistics

ARMS = ("base", "arm_a", "arm_b")
LABEL = {"base": "base", "arm_a": "arm A", "arm_b": "arm B"}
COLOUR = {"base": "#999999", "arm_a": "#1f77b4", "arm_b": "#d62728"}
SIGNED = ("leak", "targeted", "actionable", "elicits")


def goodness(key: str, value: float) -> float:
    if key == "leak":
        return -value
    if key == "length_fit":
        return -abs(value - 2.0)
    return value


def load(pattern: str) -> dict[str, dict[str, dict]]:
    out = {}
    for path in sorted(glob.glob(pattern)):
        with open(path) as handle:
            blob = json.load(handle)
        out[blob.get("rater") or os.path.basename(path)[:-5]] = {r["id"]: r for r in blob.get("labels", [])}
    return out


def consensus(raters: dict, unit_id: str, key: str) -> float | None:
    votes = [r[unit_id][key] for r in raters.values() if isinstance(r.get(unit_id, {}).get(key), int)]
    return statistics.fmean(votes) if votes else None


def by_moment(raters: dict, key: dict, dim: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for unit_id in {u for r in raters.values() for u in r}:
        entry, value = key.get(unit_id), consensus(raters, unit_id, dim)
        if entry and value is not None:
            out[entry["moment"]][entry["arm"]] = goodness(dim, value)
    return out


def paired(moments: dict, one: str, two: str) -> tuple[float, float]:
    diffs = [m[one] - m[two] for m in moments.values() if one in m and two in m]
    if len(diffs) < 3:
        return float("nan"), float("nan")
    return statistics.fmean(diffs), 1.96 * statistics.stdev(diffs) / math.sqrt(len(diffs))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", default="data/eval50/labels/agent_*.json")
    parser.add_argument("--human", default="data/eval50/labels/sophia.json")
    parser.add_argument("--key", default="data/eval50/key.json")
    parser.add_argument("--figures", default="projects/pedagogy_rm/figures")
    args = parser.parse_args()

    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    os.makedirs(args.figures, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})

    with open(args.key) as handle:
        key = json.load(handle)["key"]
    panels = {"agents": load(args.agents), "human": load(args.human)}

    # 1. GAIN OVER BASE, THREE WAYS. Levels are not comparable across panels - the raters'
    # total is a mean of 1-3 rubric scores, the probe's a mean of uncalibrated ridge outputs -
    # so the figure plots the gain, which is. A probe bar taller than the rater bars beside it
    # is the reward claiming more than anyone pays, which is what reward hacking looks like
    # before it looks like anything else.
    gains: dict[str, dict[str, float]] = {}
    for panel, raters in panels.items():
        scalar: dict[str, dict[str, float]] = collections.defaultdict(dict)
        for unit_id in {u for r in raters.values() for u in r}:
            entry = key.get(unit_id)
            parts = [goodness(d, v) for d in SIGNED if (v := consensus(raters, unit_id, d)) is not None]
            if entry and len(parts) == len(SIGNED):
                scalar[entry["moment"]][entry["arm"]] = statistics.fmean(parts)
        gains[panel] = {a: paired(scalar, a, "base")[0] for a in ("arm_a", "arm_b")}
    probe_scalar: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for entry in key.values():
        total = (entry.get("probe") or {}).get("total")
        if total is not None:
            probe_scalar[entry["moment"]][entry["arm"]] = total
    gains["probe"] = {a: paired(probe_scalar, a, "base")[0] for a in ("arm_a", "arm_b")}

    order = ["probe", "agents", "human"]
    shade = {"probe": "#d62728", "agents": "#1f77b4", "human": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    width = 0.26
    for i, panel in enumerate(order):
        xs = [j + (i - 1) * width for j in range(2)]
        ys = [gains[panel]["arm_a"], gains[panel]["arm_b"]]
        ax.bar(xs, ys, width, label=panel, color=shade[panel])
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y + 0.008, f"{y:+.2f}", ha="center", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["arm A", "arm B"])
    ax.set_ylabel("gain over base, scalarised reward")
    ax.set_title("What the reward claimed, and what the raters paid")
    ax.legend(loc="upper left", framealpha=0.95)
    ax.set_ylim(0, max(max(g.values()) for g in gains.values()) * 1.22)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "gain_three_ways.png"))
    plt.close(fig)

    # 2. THE LENGTH REGRESSION. A stacked share rather than a mean, because length_fit scores
    # 2 for right and 1 and 3 for the two ways of being wrong: an arm split evenly between
    # too-short and too-long averages exactly 2.0 and looks perfect. The two panels sit side
    # by side because they disagree about the size of it, and that disagreement is the finding.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9), sharey=True)
    for ax, (panel, raters) in zip(axes, panels.items(), strict=False):
        shares = {}
        for arm in ARMS:
            vals = [
                v
                for u, e in key.items()
                if e["arm"] == arm and (v := consensus(raters, u, "length_fit")) is not None
            ]
            if vals:
                shares[arm] = [sum(1 for v in vals if round(v) == s) / len(vals) for s in (1, 2, 3)]
        names = [a for a in ARMS if a in shares]
        bottom = [0.0] * len(names)
        for idx, (band, colour) in enumerate((("too short", "#e8a33d"), ("right", "#4c9f70"), ("too long", "#a2559c"))):
            vals = [shares[a][idx] for a in names]
            ax.bar([LABEL[a] for a in names], vals, 0.55, bottom=bottom, label=band, color=colour)
            for x, (v, b) in enumerate(zip(vals, bottom, strict=True)):
                if v > 0.06:
                    ax.text(x, b + v / 2, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
            bottom = [b + v for b, v in zip(bottom, vals, strict=True)]
        ax.set_title(f"{panel} — length_fit")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("share of turns")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), framealpha=0.95)
    fig.suptitle("Training removed over-long turns and bought it with too-short ones", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "length_split.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. PER-DIMENSION GAINS WITH INTERVALS. Signed so that right is better on every row,
    # including leak, which runs the other way on its own scale. The human's bars are shorter
    # on nothing and wider on everything: seventeen moments against sixty.
    dims = ("leak", "targeted", "actionable", "elicits", "correct", "length_fit")
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ys = range(len(dims))
    for offset, (panel, colour) in enumerate((("agents", "#1f77b4"), ("human", "#2ca02c"))):
        for arm, marker in (("arm_a", "o"), ("arm_b", "s")):
            means, errs, rows = [], [], []
            for i, dim in enumerate(dims):
                mean, half = paired(by_moment(panels[panel], key, dim), arm, "base")
                if mean == mean:
                    means.append(mean)
                    errs.append(half)
                    rows.append(i + (offset * 2 + (arm == "arm_b")) * 0.17 - 0.26)
            ax.errorbar(
                means, rows, xerr=errs, fmt=marker, ms=4, lw=0, elinewidth=1.1, capsize=2,
                color=colour, alpha=1.0 if arm == "arm_a" else 0.55,
                label=f"{panel}, {LABEL[arm]}",
            )
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(list(ys))
    ax.set_yticklabels(dims)
    ax.invert_yaxis()
    ax.set_xlabel("gain over base (signed: right is better), 95% interval over moments")
    ax.set_title("Which dimensions moved")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "dimension_gaps.png"))
    plt.close(fig)

    # 4. WHY THE AGENT LABELS WERE NOT USED TO REFIT. Measured in probe.py and agreement.py
    # and hard-coded here rather than recomputed, because the surface column needs the pooled
    # ridge fit and this script is meant to run without one. The pairing is the whole point:
    # a dimension where the grey bar is tall and the green bar is short is one the agents
    # agree about with each other and not with the person they were calibrated to.
    trust = {  # dimension: (agent-agent kappa, agent-human kappa, surface->agents, surface->human)
        "actionable": (0.76, 0.06, 0.74, 0.23),
        "elicits": (0.57, 0.06, 0.61, 0.34),
        "leak": (0.50, 0.22, 0.64, 0.28),
        "length_fit": (0.55, 0.26, 0.77, 0.72),
        "correct": (0.49, 0.27, 0.34, 0.37),
        "targeted": (0.58, 0.27, 0.08, 0.11),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
    names = list(trust)
    xs = list(range(len(names)))
    for ax, (i, j, title, ylab, left, right) in zip(
        axes,
        (
            (0, 1, "Agents agree with each other, not with the human", "weighted kappa", "agent–agent", "agent–human"),
            (2, 3, "And what they agree on is surface form", "Pearson r", "predicts agents", "predicts human"),
        ),
        strict=True,
    ):
        ax.bar([x - 0.2 for x in xs], [trust[n][i] for n in names], 0.4, label=left, color="#bbbbbb")
        ax.bar([x + 0.2 for x in xs], [trust[n][j] for n in names], 0.4, label=right, color="#2ca02c")
        ax.set_xticks(xs)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=9)
        ax.set_ylim(0, 0.85)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle("Eight features that know nothing about teaching, against six frontier models", y=1.03, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.figures, "agent_trust.png"), bbox_inches="tight")
    plt.close(fig)

    for name in ("gain_three_ways", "length_split", "dimension_gaps", "agent_trust"):
        print(f"wrote {args.figures}/{name}.png")


if __name__ == "__main__":
    main()
