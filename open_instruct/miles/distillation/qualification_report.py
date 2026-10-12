"""Summarize qualification outcomes and teaching pressure without selecting a winner."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def policy_versions(row):
    versions = {int(span["version"]) for call in row["weight_versions"] for span in call}
    if len(versions) != 1:
        raise ValueError("Synchronous evaluation requires exactly one policy version per response")
    return versions


def summarize(evaluation, metrics):
    if not evaluation:
        raise ValueError("No independent baseline evaluation was captured")
    groups = defaultdict(list)
    for row in evaluation:
        groups[row["rollout_id"], row["dataset"]].append(row)
    summaries = []
    for (step, dataset), rows in sorted(groups.items()):
        keys = [(r["sample_id"], r["sample_index"]) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate evaluation records; select one attempt before comparing")
        questions = defaultdict(list)
        for r in rows:
            questions[r["sample_id"]].append(r)
        finished = [r for r in rows if "TRUNCATED" not in r["status"].upper()]
        versions = set().union(*(policy_versions(r) for r in rows))
        if len(versions) != 1:
            raise ValueError("Evaluation contains mixed policy versions")
        summaries.append(
            {
                "rollout_id": step,
                "dataset": dataset,
                "responses": len(rows),
                "questions": len(questions),
                "accuracy": sum(r["reward"] for r in rows) / len(rows),
                "question_any_success": sum(any(r["reward"] > 0 for r in rs) for rs in questions.values())
                / len(questions),
                "samples_per_question": sorted({len(rs) for rs in questions.values()}),
                "question_sample_counts": {q: len(rs) for q, rs in questions.items()},
                "truncation_rate": 1 - len(finished) / len(rows),
                "finished_accuracy": sum(r["reward"] for r in finished) / len(finished) if finished else None,
                "mean_response_tokens": sum(r["response_length"] for r in rows) / len(rows),
                "policy_version": next(iter(versions)),
                "by_question": {q: sum(r["reward"] for r in rs) / len(rs) for q, rs in questions.items()},
            }
        )
    changes = []
    for dataset in sorted({r["dataset"] for r in summaries}):
        stages = [r for r in summaries if r["dataset"] == dataset]
        baseline = stages[0]
        if baseline["policy_version"] != 0:
            raise ValueError("First evaluation must score unchanged H045, not an already-trained checkpoint")
        for stage in stages[1:]:
            if stage["question_sample_counts"] != baseline["question_sample_counts"]:
                raise ValueError("Baseline and checkpoint evaluation must have matched question/sample coverage")
            changes.append(
                {
                    "dataset": dataset,
                    "baseline_rollout_id": baseline["rollout_id"],
                    "rollout_id": stage["rollout_id"],
                    "accuracy_change": stage["accuracy"] - baseline["accuracy"],
                    "question_changes": {q: a - baseline["by_question"][q] for q, a in stage["by_question"].items()},
                }
            )
    pressure = defaultdict(float)
    samples = 0
    for row in metrics:
        if "opd_pressure" not in row:
            continue
        samples += 1
        for k, v in row["opd_pressure"].items():
            if k.endswith("_max"):
                pressure[k] = max(pressure[k], v)
            else:
                pressure[k] += v
    return {
        "evaluation": summaries,
        "changes_from_independent_baseline": changes,
        "pressure_samples": samples,
        "pressure_totals": dict(pressure),
        "note": "First evaluation has verified policy version zero; compare coverage with the preparation manifest. "
        "Any-of-N and raw differences are descriptive, not evidence of significance.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        read_rows(args.root / "qualification-evaluation.jsonl"), read_rows(args.root / "teacher-metrics.jsonl")
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
