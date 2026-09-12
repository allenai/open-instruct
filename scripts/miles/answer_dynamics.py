"""Decompose a policy change into sharpening, moving and exploration at the answer level.

Usage: python scripts/miles/answer_dynamics.py BEFORE.jsonl AFTER.jsonl  (gsm8k_test_eval outputs)

Input: two gsm8k_test_eval JSONL files (before, after), each with 8 temperature-1
samples per question. Each question gives an empirical distribution over final
answers before and after; the change is classified and measured.
"""

import collections
import json
import math
import re
import sys

TRUNC = "<truncated>"


def answer(sample):
    if sample["finish"] == "length":
        return TRUNC
    text = re.sub(r"(\d),(\d)", r"\1\2", sample["text"])
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", text)
    return numbers[-1].lower() if numbers else "<none>"


def dist(samples):
    counts = collections.Counter(answer(s) for s in samples)
    n = sum(counts.values())
    return {a: c / n for a, c in counts.items()}


def entropy(p):
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def tv(p, q):
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in keys)


def mode(p):
    return max(p.items(), key=lambda kv: (kv[1], kv[0]))


def classify(p, q, label):
    (ma, pa), (mb, pb) = mode(p), mode(q)
    if ma == mb:
        if pb > pa + 1e-9:
            return "sharpened_correct" if mb == label else "sharpened_wrong"
        if pb < pa - 1e-9:
            return "diffused"
        return "unchanged"
    if mb == label:
        return "moved_to_correct"
    if ma == label:
        return "moved_from_correct"
    return "moved_wrong_to_wrong"


def main(before_path, after_path):
    with open(before_path) as handle:
        before = {row["id"]: row for row in map(json.loads, handle)}
    with open(after_path) as handle:
        after = {row["id"]: row for row in map(json.loads, handle)}
    rows = []
    for qid, b in before.items():
        a = after[qid]
        label = b["label"].lower()
        p, q = dist(b["sampled"]), dist(a["sampled"])
        novel = sum(v for k, v in q.items() if k not in p)  # end mass on answers never sampled at start
        abandoned = sum(v for k, v in p.items() if k not in q)
        rows.append(
            dict(
                id=qid,
                label=label,
                cls=classify(p, q, label),
                start_pass=p.get(label, 0.0),
                end_pass=q.get(label, 0.0),
                start_trunc=p.get(TRUNC, 0.0),
                end_trunc=q.get(TRUNC, 0.0),
                h_start=entropy(p),
                h_end=entropy(q),
                tv=tv(p, q),
                novel=novel,
                abandoned=abandoned,
                found=label in q and label not in p,
                lost=label in p and label not in q,
                greedy_start=b["greedy"]["correct"],
                greedy_end=a["greedy"]["correct"],
            )
        )
    n = len(rows)
    print(f"questions: {n}")

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    print("\n== change class (by answer mode and its mass), with what happened to the greedy answer")
    print(
        f"{'class':22s} {'n':>5s} {'share':>6s} {'ΔH bits':>8s} {'TV':>5s} {'novel':>6s} {'g+':>4s} {'g-':>4s} {'ΔpassR':>7s}"
    )
    classes = collections.Counter(r["cls"] for r in rows)
    for cls, _ in classes.most_common():
        g = [r for r in rows if r["cls"] == cls]
        print(
            f"{cls:22s} {len(g):5d} {len(g) / n:6.3f} {mean(r['h_end'] - r['h_start'] for r in g):8.2f} "
            f"{mean(r['tv'] for r in g):5.2f} {mean(r['novel'] for r in g):6.2f} "
            f"{sum(1 for r in g if not r['greedy_start'] and r['greedy_end']):4d} "
            f"{sum(1 for r in g if r['greedy_start'] and not r['greedy_end']):4d} "
            f"{mean(r['end_pass'] - r['start_pass'] for r in g):7.3f}"
        )

    print("\n== aggregate information measures")
    print(
        f"mean answer entropy start {mean(r['h_start'] for r in rows):.3f} bits, end {mean(r['h_end'] for r in rows):.3f} bits"
    )
    print(
        f"mean total-variation distance between start and end answer distributions: {mean(r['tv'] for r in rows):.3f}"
    )
    print(f"mean end mass on answers never sampled at start (exploration): {mean(r['novel'] for r in rows):.3f}")
    print(f"mean start mass on answers no longer sampled (abandonment): {mean(r['abandoned'] for r in rows):.3f}")
    print(
        f"questions where the correct answer was found (absent in all 8 start samples, present at end): {sum(r['found'] for r in rows)}"
    )
    print(
        f"questions where the correct answer was lost (present at start, absent at end): {sum(r['lost'] for r in rows)}"
    )

    print("\n== by start difficulty (correct samples of 8 at start): where the gradient signal was")
    print(
        f"{'start k/8':>10s} {'n':>5s} {'end pass':>9s} {'Δpass':>7s} {'ΔH':>6s} {'TV':>5s} {'novel':>6s} {'found':>6s} {'sharp':>6s} {'moved+':>7s} {'moved-':>7s} {'trunc0':>7s}"
    )
    for k in range(9):
        g = [r for r in rows if round(r["start_pass"] * 8) == k]
        if not g:
            continue
        print(
            f"{k:>10d} {len(g):5d} {mean(r['end_pass'] for r in g):9.3f} {mean(r['end_pass'] - r['start_pass'] for r in g):7.3f} "
            f"{mean(r['h_end'] - r['h_start'] for r in g):6.2f} {mean(r['tv'] for r in g):5.2f} {mean(r['novel'] for r in g):6.2f} "
            f"{sum(r['found'] for r in g):6d} {sum(r['cls'].startswith('sharpened') for r in g):6d} "
            f"{sum(r['cls'] == 'moved_to_correct' for r in g):7d} {sum(r['cls'] == 'moved_from_correct' for r in g):7d} "
            f"{mean(r['start_trunc'] for r in g):7.2f}"
        )

    print("\n== truncation as a confound")
    clean = [r for r in rows if r["start_trunc"] == 0 and r["end_trunc"] == 0]
    print(f"questions with no truncated sample at either checkpoint: {len(clean)}")
    print(
        f"  mean Δpass {mean(r['end_pass'] - r['start_pass'] for r in clean):.3f}, ΔH {mean(r['h_end'] - r['h_start'] for r in clean):.2f} bits, TV {mean(r['tv'] for r in clean):.3f}, novel {mean(r['novel'] for r in clean):.3f}"
    )
    cnt = collections.Counter(r["cls"] for r in clean)
    print("  classes:", dict(cnt.most_common()))
    with open("answer_dynamics_rows.json", "w") as handle:
        json.dump(rows, handle)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
