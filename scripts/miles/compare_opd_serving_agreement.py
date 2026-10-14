"""Compare frozen token scores and paired greedy correctness, keeping scopes separate."""

import argparse
import importlib
import json
import signal
from collections import defaultdict
from pathlib import Path

import numpy as np


def difference(reference, candidate):
    a, b = np.asarray(reference, dtype=np.float64), np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Finite, aligned token vectors are required")
    d = a - b
    return dict(
        tokens=len(a),
        abs_mean=float(np.mean(abs(d))),
        abs_p95=float(np.quantile(abs(d), 0.95)),
        abs_max=float(np.max(abs(d))),
        signed_mean_reference_minus_candidate=float(np.mean(d)),
        ratio_outside_08_128_fraction=float(np.mean((d < np.log(0.8)) | (d > np.log(1.28)))),
    )


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def timeout(*_):
    raise TimeoutError("Math verifier exceeded 45 seconds; do not count as incorrect")


def compare(panel, outputs, grade=False):
    scores = json.loads((panel / "score-panel.json").read_text())
    by_id = {r["id"]: r for r in scores}
    eval_rows = json.loads((panel / "eval-panel.json").read_text())
    eval_ids = {r["id"]: r for r in eval_rows}
    result = {
        "scope": "Frozen prefill scores vs actual first-update Megatron; HF reference is not distributed DeepSpeed.",
        "score_comparisons": {},
        "repeatability": {},
        "accuracy": {},
        "paired_accuracy": {},
    }
    reference = [x for r in scores for x in r["trainer_logprobs"]]
    result["original_decode_vs_megatron"] = difference(
        reference, [x for r in scores for x in r["original_rollout_logprobs"]]
    )
    candidates = {}
    accuracy = {}
    verifier = importlib.import_module("open_instruct.ground_truth_utils").MathVerifier() if grade else None
    if grade:
        signal.signal(signal.SIGALRM, timeout)
    for name, path in outputs.items():
        if not (path / "complete.json").is_file():
            raise ValueError("Incomplete arm: " + name)
        waves = defaultdict(dict)
        for row in load_jsonl(path / "scores.jsonl"):
            if row["id"] in waves[row["wave"]]:
                raise ValueError("Duplicate scored sequence")
            if row["id"] not in by_id or len(row["logprobs"]) != by_id[row["id"]]["response_length"]:
                raise ValueError("Score IDs or token lengths differ")
            waves[row["wave"]][row["id"]] = row["logprobs"]
        for wave, values in waves.items():
            if values.keys() != by_id.keys():
                raise ValueError("Missing scored sequences")
            flat = [v for row in scores for v in values[row["id"]]]
            candidates[name + "/" + wave] = flat
            result["score_comparisons"][name + "/" + wave] = difference(reference, flat)
        generations = path / "generations.jsonl"
        if not generations.exists():
            continue
        gen = load_jsonl(generations)
        first = {r["id"]: r for r in gen if r["wave"] == "repeat-r0"}
        summaries = defaultdict(lambda: dict(requests=0, changed=0))
        for r in gen:
            if r["wave"] == "accuracy":
                continue
            s = summaries[r["wave"]]
            s["requests"] += 1
            s["changed"] += int(r["output_ids"] != first[r["id"]]["output_ids"])
        result["repeatability"][name] = dict(summaries)
        acc = [r for r in gen if r["wave"] == "accuracy"]
        if len(acc) != len(eval_ids) or {r["id"] for r in acc} != eval_ids.keys():
            raise ValueError("Incomplete accuracy panel")
        if grade:
            graded = {}
            for r in acc:
                q = eval_ids[r["id"]]
                try:
                    signal.alarm(45)
                    reward = verifier([], r["response"], str(q["label"]), q["prompt"]).score
                finally:
                    signal.alarm(0)
                graded[r["id"]] = dict(correct=bool(reward), length=len(r["output_ids"]), finish=r["finish"])
            accuracy[name] = graded
            result["accuracy"][name] = {
                dataset: dict(
                    n=sum(q["dataset"] == dataset for q in eval_rows),
                    correct=sum(graded[q["id"]]["correct"] for q in eval_rows if q["dataset"] == dataset),
                    mean_tokens=float(
                        np.mean([graded[q["id"]]["length"] for q in eval_rows if q["dataset"] == dataset])
                    ),
                )
                for dataset in sorted({q["dataset"] for q in eval_rows})
            }
    result["serving_self_comparisons"] = {}
    for name in outputs:
        keys = [k for k in candidates if k.startswith(name + "/")]
        for key in keys[1:]:
            result["serving_self_comparisons"][keys[0] + " vs " + key] = difference(
                candidates[keys[0]], candidates[key]
            )
    if "hf/hf-r0" in candidates:
        result["vs_hf_forward_reference"] = {
            k: difference(candidates["hf/hf-r0"], v) for k, v in candidates.items() if not k.startswith("hf/")
        }
    if "sglang-baseline" in accuracy:
        a = accuracy["sglang-baseline"]
        for name, b in accuracy.items():
            if name == "sglang-baseline":
                continue
            result["paired_accuracy"][name] = {
                dataset: dict(
                    right_to_wrong=sum(
                        a[q["id"]]["correct"] and not b[q["id"]]["correct"]
                        for q in eval_rows
                        if q["dataset"] == dataset
                    ),
                    wrong_to_right=sum(
                        not a[q["id"]]["correct"] and b[q["id"]]["correct"]
                        for q in eval_rows
                        if q["dataset"] == dataset
                    ),
                )
                for dataset in sorted({q["dataset"] for q in eval_rows})
            }
    result["graded_questions"] = accuracy
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--arm", action="append", required=True, help="NAME=OUTPUT_DIRECTORY")
    parser.add_argument("--grade", action="store_true")
    options = parser.parse_args()
    outputs = {k: Path(v) for k, v in (x.split("=", 1) for x in options.arm)}
    print(json.dumps(compare(options.panel, outputs, options.grade), indent=2))


if __name__ == "__main__":
    main()
