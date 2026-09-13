"""Render measured cycle comparisons and sampled queues as standalone report figures.

Usage: python -m scripts.miles.plot_throughput_basket results.json output-directory
Optional --run-root case=/downloaded/run adds continuous occupancy panels.
This is an offline reporting dependency; matplotlib is not needed by training.
"""

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

matplotlib.use("Agg")
COLORS = {"generation_wait": "#e5a545", "training": "#397f9d", "publication": "#7c6ba6"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def comparisons(data, output):
    entries = [
        (name, row)
        for name, row in data.items()
        if "driver_timings" in row and row["analysis"]["measured_updates"] >= 6
    ]
    names = [name.replace("small-", "").replace("steady-", "") for name, _ in entries]
    fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, len(entries) * 0.55)), layout="constrained")
    offsets = [0.0] * len(entries)
    for stage, color in COLORS.items():
        values = [statistics.mean(r[stage + "_seconds"] for r in row["analysis"]["per_update"]) for _, row in entries]
        axes[0].barh(names, values, left=offsets, color=color, label=stage.replace("_", " "))
        offsets = [a + b for a, b in zip(offsets, values)]
    axes[0].set_xlabel("Mean seconds / awaited driver cycle")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3, fontsize=8)
    axes[0].set_title("Trainer's awaited cycle")
    axes[1].barh(names, [row["analysis"]["useful_response_tokens_per_second"] for _, row in entries], color="#397f9d")
    axes[1].set_xlabel("Consumed response tokens / cycle second")
    axes[1].set_title("Useful throughput")
    axes[2].barh(names, [100 * row["analysis"]["discarded_token_fraction"] for _, row in entries], color="#bd6657")
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel("% of dequeued response tokens discarded")
    axes[2].set_title("Waste at the completed queue")
    for ax in axes:
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Measured warm windows • generation overlaps training • lifecycle qualification listed separately")
    save(fig, output, "cycle-comparison")

    fig, axes = plt.subplots(len(entries), 3, figsize=(14, 2.2 * len(entries)), squeeze=False, layout="constrained")
    prefix = "rollout/fully_async/completed_queue/"
    for (name, row), ax in zip(entries, axes):
        timings = row["training_timings_rank0"]
        for event, field, label in (
            ("optimizer", "elapsed_seconds", "forward/backward/optimizer"),
            ("score_timing", "seconds", "scoring"),
        ):
            records = [r for r in timings if r["event"] == event]
            ax[0].plot(
                [r.get("rollout_id", r.get("step", 1) - 1) + 1 for r in records],
                [r[field] for r in records],
                marker=".",
                label=label,
            )
        ax[0].set_yscale("log")
        ax[0].set_ylabel(name + "\nseconds (log scale)", fontsize=8)
        flow = row["flow"]
        steps = [r["rollout_id"] + 1 for r in flow]
        ax[1].plot(
            steps,
            [r["queue_metrics"].get("rollout/fully_async/queue_size", float("nan")) for r in flow],
            marker=".",
            color="#397f9d",
        )
        ax[2].plot(
            steps,
            [100 * r["queue_metrics"].get(prefix + "dropped_response_tokens_fraction", 0) for r in flow],
            marker=".",
            color="#bd6657",
        )
        ax[2].set_ylim(0, 100)
        for a in ax:
            a.axvspan(0.5, row["analysis"]["warmup_updates"] + 0.5, color="#bbbbbb", alpha=0.15)
            a.grid(alpha=0.2)
            a.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_title("Training components; shaded warmup excluded")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("Completed groups after collection (instantaneous)")
    axes[0, 2].set_title("Discarded response tokens (%)")
    for a in axes[-1]:
        a.set_xlabel("Optimizer update (1-based)")
    save(fig, output, "warmup-and-queue-traces")

    labels = ["0–255", "256–511", "512–1023", "1024–2047", "2048–4095", "4096–8191"]
    bins = [s.replace("–", "_") for s in labels]
    matrix, counts = [], []
    for _, row in entries:
        values, totals = [], []
        for label in bins:
            chosen = row["flow"][row["analysis"]["warmup_updates"] :]
            dropped = sum(r["queue_metrics"].get(prefix + "dropped_samples_by_length/" + label, 0) for r in chosen)
            delivered = sum(r["queue_metrics"].get(prefix + "delivered_samples_by_length/" + label, 0) for r in chosen)
            values.append(dropped / (dropped + delivered) if dropped + delivered else float("nan"))
            totals.append((dropped, dropped + delivered))
        matrix.append(values)
        counts.append(totals)
    fig, ax = plt.subplots(figsize=(10, max(3, len(entries) * 0.6)), layout="constrained")
    heat = ax.imshow(matrix, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Response length (tokens); cell = dropped / dequeued samples")
    ax.set_title("Length-selective loss at the completed queue • measured warm windows")
    for i, row in enumerate(counts):
        for j, (dropped, total) in enumerate(row):
            value = dropped / total if total else 0
            ax.text(
                j,
                i,
                f"{dropped}/{total}" if total else "no samples",
                ha="center",
                va="center",
                color="white" if value > 0.6 else "black",
                fontsize=9,
            )
    fig.colorbar(heat, ax=ax, label="Fraction discarded")
    save(fig, output, "discard-by-length")


def observed_steps(points, *, max_hold_minutes=10 / 60):
    """Break sampled lines at missing observations instead of implying long holds."""
    result = []
    for i, (timestamp, value) in enumerate(points):
        result.append((timestamp, float("nan") if value is None else value))
        if i + 1 < len(points) and points[i + 1][0] > timestamp + max_hold_minutes:
            result.append((timestamp + max_hold_minutes, float("nan")))
    return [p[0] for p in result], [p[1] for p in result]


def occupancy(root, output, name):
    stages = read_rows(root / "checkpoints/driver_timing.jsonl")
    producer = read_rows(root / "checkpoints/pipeline_occupancy.jsonl")
    engines = sorted(
        (row for path in (root / "checkpoints").glob("engine_occupancy*.jsonl") for row in read_rows(path)),
        key=lambda row: row["time_unix"],
    )
    start = min(r["time_unix"] for r in producer)
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True, layout="constrained")
    for stage, color in {**COLORS, "final_generation_drain": "#bcbcbc"}.items():
        intervals = [((r["started_unix"] - start) / 60, r["seconds"] / 60) for r in stages if r["stage"] == stage]
        axes[0].broken_barh(intervals, (0, 1), facecolors=color, label=stage.replace("_", " "))
    axes[0].set_yticks([])
    axes[0].set_ylabel("Trainer driver")
    axes[0].legend(ncols=3, fontsize=8)
    for key, axis, label in (
        ("producer_owned_groups", 1, "Owned prompt groups"),
        ("completed_queue_groups", 1, "Completed queue groups"),
        ("completed_queue_capacity_groups", 1, "Completed queue capacity"),
        ("http_active_requests", 2, "HTTP active (includes server wait)"),
        ("http_waiting_requests", 2, "Waiting for HTTP admission"),
        ("http_capacity_requests", 2, "HTTP capacity"),
    ):
        points = [((r["time_unix"] - start) / 60, r.get(key)) for r in producer]
        axes[axis].step(*observed_steps(points), where="post", label=label)
    for identity in sorted({r["engine"] for r in engines if "engine" in r}):
        for key, axis in (("num_running_reqs", 3), ("num_queue_reqs", 4)):
            points = []
            for r in engines:
                if r.get("engine") != identity:
                    continue
                series = [s for s in r.get("series", []) if s["name"] == key]
                # These exercises use one TP1/DP1 engine per URL. Multi-series
                # endpoints need an explicit rank aggregation rather than a guess.
                value = series[0]["value"] if len(series) == 1 else None
                points.append(((r["time_unix"] - start) / 60, value))
            if any(value is not None for _, value in points):
                axes[axis].step(*observed_steps(points), where="post", label=identity.removeprefix("http://"))
    for index, ax in enumerate(axes[1:], 1):
        ax.set_ylabel(("", "Prompt groups", "Requests", "Engine running", "Engine waiting")[index])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
        if 0 < len(ax.lines) <= 8:
            ax.legend(ncols=4, fontsize=7)
        elif ax.lines:
            ax.text(0.01, 0.95, f"{len(ax.lines)} individual engines", transform=ax.transAxes, va="top", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No usable observations", transform=ax.transAxes, ha="center")
    measured_path = root / "measured-cycles.json"
    warmup = json.loads(measured_path.read_text())["warmup_updates"] if measured_path.exists() else 6
    measured_stages = [r for r in stages if r["stage"] == "generation_wait" and r["rollout_id"] == warmup]
    warm_start = (measured_stages[0]["started_unix"] - start) / 60 if measured_stages else None
    if warm_start is not None:
        for ax in axes:
            ax.axvline(warm_start, linestyle="--", color="#333333", linewidth=1)
        axes[0].text(warm_start, 1.02, "Measured window begins", fontsize=8, ha="right")
    axes[-1].set_xlabel("Minutes since producer observation began (includes startup and shutdown)")
    fig.suptitle(name + " • sampled occupancy, not hardware GPU utilization")
    save(fig, output, name + "-pipeline")

    paths = sorted((root / "checkpoints").glob("gpu_usage_node*.jsonl"))
    if not paths:
        return
    end = max(r["time_unix"] for r in producer)
    grid = np.arange(start, end + 5, 5)
    traces, labels = [], []
    for path in paths:
        records = read_rows(path)
        devices = sorted({(d["index"], d["uuid"]) for r in records for d in r.get("devices", [])}, key=lambda d: d[0])
        for index, identity in devices:
            trace = np.full(len(grid), np.nan)
            for position, record in enumerate(records):
                matches = [d for d in record.get("devices", []) if d["uuid"] == identity]
                if len(matches) != 1 or matches[0]["utilization.gpu"] is None:
                    continue
                t = record["time_unix"]
                next_time = records[position + 1]["time_unix"] if position + 1 < len(records) else end
                trace[(grid >= t) & (grid < min(t + 10, next_time))] = matches[0]["utilization.gpu"]
            traces.append(trace)
            labels.append(f"node {path.stem.split('node')[-1]} · GPU {int(index)}")
    if not traces:
        return
    fig, ax = plt.subplots(figsize=(13, max(4, len(traces) * 0.2)), layout="constrained")
    color_map = plt.get_cmap("Blues").copy()
    color_map.set_bad("#dddddd")
    heat = ax.imshow(
        traces,
        aspect="auto",
        interpolation="nearest",
        cmap=color_map,
        vmin=0,
        vmax=100,
        extent=(0, (grid[-1] - start) / 60, len(traces) - 0.5, -0.5),
    )
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Minutes since producer observation began")
    ax.set_title(name + " • NVML GPU activity • gray means unobserved")
    fig.colorbar(heat, ax=ax, label="GPU activity (%)")
    save(fig, output, name + "-gpu-activity")


def pipeline_map(report, output, name):
    """Annotate the data path with measured queue occupancy and driver service times."""
    if "occupancy" not in report:
        return
    occupancy_data = report["occupancy"]
    queues = occupancy_data["pipeline"]

    def value(key, field="mean", suffix=""):
        row = queues.get(key, {})
        number = row.get(field)
        if number is None:
            return "unobserved"
        return f"{number:.1f}{suffix}"

    def engine_value(key):
        rows = [s for e in occupancy_data["engines"].values() for s in e["series"] if s["name"] == key]
        numbers = [s["mean"] for s in rows if s["mean"] is not None]
        return f"{min(numbers):.1f}–{max(numbers):.1f} / engine" if numbers else "unobserved"

    means = {stage: statistics.mean(r[stage + "_seconds"] for r in report["per_update"]) for stage in COLORS}
    boxes = [
        (
            0,
            1,
            "Producer",
            f"Owned groups: {value('producer_owned_groups')} mean\nActive tasks: {value('producer_active_group_tasks')} mean",
        ),
        (
            1,
            1,
            "HTTP admission",
            f"Active: {value('http_active_requests')} mean\nWaiting: {value('http_waiting_requests')} mean\nCapacity: {value('http_capacity_requests')} requests",
        ),
        (2, 1, "Engine queue", f"Waiting: {engine_value('num_queue_reqs')}\nRouter residence: not timed"),
        (3, 1, "Prefill / decode", f"Running: {engine_value('num_running_reqs')}\nSee NVML activity timeline"),
        (
            3,
            0,
            "Verification / siblings",
            "Local GSM8K verification\nWait for all group members\nNot separately timed",
        ),
        (
            2,
            0,
            "Completed FIFO",
            f"Groups: {value('completed_queue_groups')} mean\nP95: {value('completed_queue_groups', 'p95')} / capacity {value('completed_queue_capacity_groups')}\nDropped tokens: {100 * report['discarded_token_fraction']:.1f}%",
        ),
        (
            1,
            0,
            "Trainer",
            f"Batch wait: {means['generation_wait']:.1f} s / cycle\nScore + train: {means['training']:.1f} s / cycle\nTraining + publication: {100 * (1 - report['trainer_wait_fraction']):.1f}% of cycle",
        ),
        (
            0,
            0,
            "Weight publication",
            f"{means['publication']:.1f} s / cycle\nRefresh paused engines\nRetained prefix is re-prefilled",
        ),
    ]
    fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
    ax.set(xlim=(-0.65, 3.65), ylim=(-0.55, 1.6))
    ax.axis("off")
    for x, y, title, body in boxes:
        ax.text(
            x,
            y,
            title + "\n\n" + body,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#eef3f5", edgecolor="#397f9d"),
        )
    for left, right in zip(boxes, boxes[1:]):
        x1, y1 = left[:2]
        x2, y2 = right[:2]
        dx, dy = x2 - x1, y2 - y1
        ax.annotate(
            "",
            xy=(x2 - 0.43 * dx, y2 - 0.28 * dy),
            xytext=(x1 + 0.43 * dx, y1 + 0.28 * dy),
            arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.5),
        )
    ax.text(
        1.5,
        -0.42,
        "Publication updates the serving fleet; the generation path runs concurrently with training.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    coverage = [r["coverage_fraction"] for r in queues.values() if r["mean"] is not None]
    minimum = min(coverage) if coverage else 0
    fig.suptitle(
        f"{name} • {report['measured_updates']} measured updates • queue coverage ≥ {100 * minimum:.1f}%\n"
        "Queue means are time-weighted; driver fractions are not hardware utilization",
        fontsize=13,
    )
    save(fig, output, name + "-map")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--run-root", action="append", default=[], help="case=/path/to/downloaded/run")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    comparisons(json.loads(args.results.read_text()), args.output)
    for item in args.run_root:
        name, root = item.split("=", 1)
        occupancy(Path(root), args.output, name)
        measured = Path(root) / "measured-cycles.json"
        if measured.exists():
            pipeline_map(json.loads(measured.read_text()), args.output, name)


if __name__ == "__main__":
    main()
