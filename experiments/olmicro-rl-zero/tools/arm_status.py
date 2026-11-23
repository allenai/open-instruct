"""One-screen health summary of a running MILES RL run from its output root and a saved `beaker job logs` file.

Usage: python arm_status.py RUN_ROOT JOB_LOG [TOTAL_UPDATES]
"""

import collections
import glob
import json
import re
import statistics
import sys
from pathlib import Path


def main():
    root, log = Path(sys.argv[1]), Path(sys.argv[2])
    total = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    flow = [json.loads(line) for line in (root / "checkpoints/rollout_flow.jsonl").read_text().splitlines()]
    timing = [json.loads(line) for line in (root / "checkpoints/driver_timing.jsonl").read_text().splitlines()]
    text = log.read_text(errors="replace")

    train = {r["rollout_id"]: r for r in timing if r["stage"] == "training" and r.get("rollout_id") is not None}
    gen = [r["seconds"] for r in timing if r["stage"] == "generation_wait"]
    ids = sorted(train)
    print(f"updates done: {len(flow)}/{total}")
    if len(ids) > 11:
        a, b = train[ids[-11]], train[ids[-1]]
        per = (b["started_unix"] + b["seconds"] - a["started_unix"]) / (ids[-1] - ids[-11])
        print(f"wall {per:.1f} s/update (last 10); train median {statistics.median(train[i]['seconds'] for i in ids[-10:]):.1f}s; "
              f"generation wait median {statistics.median(gen[-10:]):.1f}s; remaining ~{(total - len(flow)) * per / 3600:.2f} h + evals")

    cap = max(max(r["response_lengths"]) for r in flow)
    step = 50
    for start in range(0, len(flow), step):
        seg = flow[start : start + step]
        rew = statistics.mean(r["queue_metrics"].get("rollout/raw_reward_unfiltered", 0) for r in seg)
        ln = statistics.mean(statistics.mean(r["response_lengths"]) for r in seg)
        tr = statistics.mean(sum(x >= 7680 for x in r["response_lengths"]) / len(r["response_lengths"]) for r in seg)
        print(f"  updates {start:3d}-{start + len(seg) - 1:3d}: reward {rew:.3f}  mean_len {ln:5.0f}  trunc@7680 {tr:.3f}")
    print(f"  longest response so far: {cap}")

    evals = collections.defaultdict(list)
    for name, value in re.findall(r"'eval/(math500|gsm8k_platinum)/episode_raw_reward': ([-0-9.e]+)", text):
        evals[name].append(round(100 * float(value), 1))
    print("evals (every 50):", dict(evals))

    grads = [float(v) for v in re.findall(r"'train/grad_norm': ([-0-9.e]+)", text)]
    diffs = [float(v) for v in re.findall(r"'train/train_rollout_logprob_abs_diff': ([-0-9.e]+)", text)]
    checks = collections.Counter(re.findall(r"\[check_tensors\] ([a-z_ ]+?):", text))
    if grads:
        print(f"grad_norm n={len(grads)} min={min(grads):.3g} max={max(grads):.3g} zeros={sum(g == 0 for g in grads)}; "
              f"logprob diff median={statistics.median(diffs):.4f} max={max(diffs):.4f}")
    print(f"weight checks: {dict(checks)}; tracebacks={text.count('Traceback')}; nan/inf words={len(re.findall(r'\bnan\b|\binf\b', text, re.I))}")

    files = glob.glob(f"/weka/oe-adapt-default/jacobm/olmicro/inference-records/**/{root.name}-*/records-*.jsonl", recursive=True)
    groups = []
    for path in files:
        for line in open(path):
            try:
                group = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "responses" in group:
                groups.append(group)
    if groups:
        mixed = [0 < sum(x["reward"] for x in g["responses"]) < len(g["responses"]) for g in groups]
        third = max(1, len(groups) // 3)
        print(f"records: {len(groups)} groups; mixed fraction by thirds "
              f"{[round(statistics.mean(mixed[i * third:(i + 1) * third]), 3) for i in range(3)]}; "
              f"all-correct {statistics.mean(all(x['reward'] >= 1 for x in g['responses']) for g in groups):.3f}")


if __name__ == "__main__":
    main()
