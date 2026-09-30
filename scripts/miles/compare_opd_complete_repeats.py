"""Grade complete repeated answers; report numerical and correctness stability separately."""

import argparse
import importlib
import json
import signal
from collections import defaultdict
from pathlib import Path

from scripts.miles import compare_opd_serving_agreement


def summarize(rows, questions, grades):
    by_id = {r["id"]: r for r in questions}
    reference = {r["id"]: r for r in rows if r["wave"] == "repeat-r0"}
    if len(reference) != 8 or reference.keys() != by_id.keys():
        raise ValueError("Expected exactly eight reference questions")
    waves = defaultdict(list)
    for row in rows:
        if row["id"] not in reference:
            raise ValueError("Unexpected question")
        waves[row["wave"]].append(row)
    if set(waves) != {"repeat-r0", "repeat-r1", "reversed", "single", "crowded"}:
        raise ValueError("Missing comparison wave")
    result = {}
    correctness_by_question = defaultdict(set)
    for wave, values in waves.items():
        expected = 2 if wave == "crowded" else 1
        if any(sum(r["id"] == q for r in values) != expected for q in reference):
            raise ValueError("Missing or duplicated requests")
        summary = dict(
            requests=len(values), token_changed=0, right_to_wrong=0, wrong_to_right=0, correct=0, truncated=0
        )
        for row in values:
            base = reference[row["id"]]
            a, b = grades[(base["id"], base["response"])], grades[(row["id"], row["response"])]
            summary["token_changed"] += row["output_ids"] != base["output_ids"]
            summary["right_to_wrong"] += a and not b
            summary["wrong_to_right"] += not a and b
            summary["correct"] += b
            summary["truncated"] += row["finish"] == "length"
            correctness_by_question[row["id"]].add(b)
        result[wave] = summary
    return dict(
        waves=result,
        questions_with_variable_correctness=sorted(
            q for q, outcomes in correctness_by_question.items() if len(outcomes) > 1
        ),
        scope="Eight enriched questions, not an accuracy estimate; crowded requests repeat each question twice.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True)
    args = parser.parse_args()
    questions = json.loads((args.panel / "eval-panel.json").read_text())
    by_id = {q["id"]: q for q in questions}
    verifier = importlib.import_module("open_instruct.ground_truth_utils").MathVerifier()
    signal.signal(signal.SIGALRM, compare_opd_serving_agreement.timeout)
    grades, result = {}, {}
    for name, directory in (item.split("=", 1) for item in args.arm):
        directory = Path(directory)
        if not (directory / "complete.json").is_file():
            raise ValueError("Incomplete arm: " + name)
        rows = compare_opd_serving_agreement.load_jsonl(directory / "generations.jsonl")
        for row in rows:
            key = (row["id"], row["response"])
            if key not in grades:
                question = by_id[row["id"]]
                try:
                    signal.alarm(45)
                    grades[key] = bool(verifier([], row["response"], str(question["label"]), question["prompt"]).score)
                finally:
                    signal.alarm(0)
        result[name] = summarize(rows, questions, grades)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
