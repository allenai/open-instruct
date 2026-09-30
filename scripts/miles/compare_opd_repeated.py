"""Compare repeated evaluations using paired questions as the uncertainty unit."""

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def read(path):
    rows = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        key = (row["dataset"], row["question_id"], row["mode"], row["repeat"])
        if key in rows:
            raise ValueError(f"Duplicate response: {key}")
        rows[key] = row
    if not rows:
        raise ValueError("Empty evaluation")
    return rows


def summarize(left, right):
    if left.keys() != right.keys():
        raise ValueError("Arms must contain exactly the same questions and repetitions")
    groups = defaultdict(lambda: defaultdict(list))
    for key in sorted(left):
        a, b = left[key], right[key]
        for field in ("prompt", "label", "input_ids", "seed"):
            if a[field] != b[field]:
                raise ValueError(f"Paired {field} differs: {key}")
        groups[(key[0], key[2])][key[1]].append((a, b))
    result = {}
    for (dataset, mode), questions in groups.items():
        sizes = {len(pairs) for pairs in questions.values()}
        if len(sizes) != 1:
            raise ValueError("Unequal repetition counts across questions")
        repeats = sizes.pop()
        differences = []
        sampling_variances = []
        flattened = []
        for pairs in questions.values():
            deltas = [int(b["reward"] > 0) - int(a["reward"] > 0) for a, b in pairs]
            differences.append(statistics.mean(deltas))
            if repeats > 1:
                sampling_variances.append(statistics.variance(deltas) / repeats)
            flattened.extend(pairs)
        rng = random.Random(42)
        draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(5000))
        entry = {
            "questions": len(questions),
            "responses_per_arm": len(flattened),
            "repetitions": repeats,
            "accuracy_delta_right_minus_left": statistics.mean(differences),
            "paired_question_bootstrap_95_interval": [draws[124], draws[4874]],
            "conditional_sampling_se_of_delta": math.sqrt(sum(sampling_variances)) / len(questions)
            if sampling_variances
            else None,
            "question_deltas": dict(zip(questions, differences, strict=True)),
        }
        for name, index in (("left", 0), ("right", 1)):
            rows = [pair[index] for pair in flattened]
            entry[name] = {
                "pass_at_1": statistics.mean(row["reward"] > 0 for row in rows),
                "mean_response_tokens": statistics.mean(row["response_length"] for row in rows),
                "truncated_fraction": statistics.mean(row["status"] == "truncated" for row in rows),
            }
        result[f"{dataset}/{mode}"] = entry
    return result


def compare(paths):
    arms = {name: read(path / "responses.jsonl") for name, path in paths.items()}
    for name, path in paths.items():
        expected = json.loads((path / "provenance.json").read_text())["expected_responses"]
        complete = json.loads((path / "complete.json").read_text())
        if len(arms[name]) != expected or complete["responses"] != expected:
            raise ValueError(f"Incomplete arm: {name}")
    return {
        "scope": "Sampled pass@1 is mean correctness, not pass@16. Bootstrap resamples paired questions, not individual answers; it excludes training-seed variability. Pairwise intervals are unadjusted and exploratory.",
        "comparisons": {
            f"{left}__vs__{right}": summarize(arms[left], arms[right])
            for left, right in itertools.combinations(arms, 2)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, help="NAME=RESULT_DIRECTORY")
    args = parser.parse_args()
    paths = {name: Path(path) for name, path in (entry.split("=", 1) for entry in args.arm)}
    print(json.dumps(compare(paths), indent=2))
