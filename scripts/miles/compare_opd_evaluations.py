"""Pair captured greedy evaluations by dataset and exact prompt, not completion order."""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def read(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        prompt = json.dumps(row["prompt"], sort_keys=True, ensure_ascii=False)
        key = (row["dataset"], hashlib.sha256(prompt.encode()).hexdigest())
        if key in rows:
            raise ValueError("Expected one greedy response per dataset/prompt")
        rows[key] = row
    return rows


def correlation(x, y):
    n = len(x)
    x_mean, y_mean = sum(x) / n, sum(y) / n
    x_centered = [value - x_mean for value in x]
    y_centered = [value - y_mean for value in y]
    denominator = math.sqrt(sum(v * v for v in x_centered) * sum(v * v for v in y_centered))
    return sum(a * b for a, b in zip(x_centered, y_centered, strict=True)) / denominator if denominator else None


def summarize(pairs):
    differences = [int(right["reward"] > 0) - int(left["reward"] > 0) for left, right in pairs]
    length_differences = [right["response_length"] - left["response_length"] for left, right in pairs]
    n = len(pairs)
    rng = random.Random(42)
    bootstraps = sorted(sum(rng.choices(differences, k=n)) / n for _ in range(2000))
    groups = defaultdict(list)
    for left, right in pairs:
        outcome = (bool(left["reward"] > 0), bool(right["reward"] > 0))
        groups[outcome].append((left, right))
    result = {
        "questions": n,
        "accuracy_delta_right_minus_left": sum(differences) / n,
        "paired_question_bootstrap_95_interval": [bootstraps[49], bootstraps[1949]],
        "length_delta_mean_right_minus_left": sum(length_differences) / n,
        "length_delta_accuracy_delta_correlation": correlation(length_differences, differences),
        "outcomes": {},
    }
    for side, index in [("left", 0), ("right", 1)]:
        rows = [pair[index] for pair in pairs]
        result[side] = {
            "accuracy": sum(row["reward"] > 0 for row in rows) / n,
            "mean_response_tokens": sum(row["response_length"] for row in rows) / n,
            "natural_stop_fraction": sum(row["status"] == "completed" for row in rows) / n,
            "truncated_fraction": sum(row["status"] == "truncated" for row in rows) / n,
            "correct_while_truncated": sum(row["reward"] > 0 and row["status"] == "truncated" for row in rows),
            "boxed_answer_fraction": sum("\\boxed{" in row["response"] for row in rows) / n,
        }
    for outcome, label in [
        ((True, True), "both_correct"),
        ((False, False), "both_wrong"),
        ((False, True), "right_gains"),
        ((True, False), "right_losses"),
    ]:
        selected = groups[outcome]
        result["outcomes"][label] = {
            "questions": len(selected),
            "mean_length_delta": sum(b["response_length"] - a["response_length"] for a, b in selected) / len(selected)
            if selected
            else None,
            "right_longer": sum(b["response_length"] > a["response_length"] for a, b in selected),
            "left_truncated_right_stopped": sum(
                a["status"] == "truncated" and b["status"] == "completed" for a, b in selected
            ),
            "left_stopped_right_truncated": sum(
                a["status"] == "completed" and b["status"] == "truncated" for a, b in selected
            ),
        }
    return result


def compare(left_path, right_path):
    left, right = read(left_path), read(right_path)
    if left.keys() != right.keys():
        raise ValueError("Evaluations must contain exactly the same dataset/prompt keys")
    if not left:
        raise ValueError("Empty evaluations")
    datasets = defaultdict(list)
    for key in sorted(left):
        if left[key]["label"] != right[key]["label"]:
            raise ValueError("Ground truth differs for a paired prompt")
        datasets[key[0]].append((left[key], right[key]))
    return {
        "scope": "Paired questions from single greedy evaluations; bootstrap excludes training-seed variation. Correlation is descriptive, not causal.",
        "sources": {
            side: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for side, path in [("left", left_path), ("right", right_path)]
        },
        "datasets": {name: summarize(pairs) for name, pairs in datasets.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.left, args.right), indent=2))


if __name__ == "__main__":
    main()
