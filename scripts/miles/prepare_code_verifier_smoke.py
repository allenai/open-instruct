"""Prepare a small, pinned Dolci coding sample and qualify the new HTTP verifier.

This is an integration run, not a dataset-quality or learning benchmark. Preserve
complete test suites, exclude source solutions from prompts, and record selected
source IDs. A reference solution is used only for preflight grading controls.
The output directory must be new; existing experiment inputs are never replaced.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import datasets
from code_execution_client import CodeExecutionClient
from code_execution_service import protocol
from transformers import AutoTokenizer

DATASET = "allenai/Dolci-RLZero-Code-7B"
REVISION = "054c0a1f9c5fd52566ad9124aa2e702924803cd3"
SERVICE_URL = "https://1n7avfskve.execute-api.us-west-2.amazonaws.com"
SERVICE_SHA256 = "4627e53f1c4901d95c6a36a5326c254ae620123b1469e5987579fcee3197726e"


def prepare_row(row, index, tokenizer, max_prompt_tokens):
    labels = row["ground_truth"]
    if not isinstance(labels, list) or len(labels) != 1:
        raise ValueError("expected one code-verifier target")
    tests = protocol.decode_tests(labels[0])
    if not tests:
        raise ValueError("empty test suite")
    stdio = isinstance(tests[0], dict)
    protocol.validate_tests("stdio" if stdio else "assert", tests)
    prompt = row["prompt"]
    if not isinstance(prompt, str) or not prompt.startswith("user: "):
        raise ValueError("unexpected prompt format")
    question = prompt.removeprefix("user: ")
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    if len(tokenizer.encode(rendered, add_special_tokens=False)) > max_prompt_tokens:
        raise ValueError("prompt exceeds the smoke-run limit")
    name = "code_stdio" if stdio else "code"
    return {
        "input": rendered,
        "label": labels[0],
        "metadata": {
            "prepared_sample_id": f"dolci-code:{REVISION}:{index}:{row['id']}",
            "source_dataset": DATASET,
            "source_revision": REVISION,
            "source_row": index,
            "source_id": row["id"],
            "query": question,
            "verifiers": [{"name": name, "target": tests, "weight": 1.0}],
        },
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--scan-rows", type=int, default=512)
    parser.add_argument("--train-count", type=int, default=32)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    token = os.environ["CODE_EXECUTION_TOKEN"]
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    source = datasets.load_dataset(DATASET, revision=REVISION, split=f"train[:{args.scan_rows}]")
    candidates, exclusions = [], {}
    for index, row in enumerate(source):
        try:
            prepared = prepare_row(row, index, tokenizer, args.max_prompt_tokens)
        except (ValueError, protocol.BadRequest) as error:
            reason = str(error)
            exclusions[reason] = exclusions.get(reason, 0) + 1
            continue
        candidates.append((prepared, row["solution"]))
    random.Random(args.seed).shuffle(candidates)
    total = args.train_count + args.eval_count
    if len(candidates) < total:
        raise ValueError(f"only {len(candidates)} eligible rows for {total} requested")
    selected = candidates[:total]
    records = [record for record, _ in selected]
    registry = {
        name: {
            "factory": "code_execution_client.verifier.CodeVerifier",
            "config": {
                "api_url": SERVICE_URL,
                "stdio": name == "code_stdio",
                "token_env": "CODE_EXECUTION_TOKEN",
                "max_execution_time": 1.0,
                "pass_rate_reward_threshold": 0.0,
                "expected_source_sha256": SERVICE_SHA256,
                "total_timeout": 300.0,
            },
        }
        for name in {record["metadata"]["verifiers"][0]["name"] for record in records}
    }
    controls = []
    client = CodeExecutionClient(SERVICE_URL, token=token, expected_source_sha256=SERVICE_SHA256)
    try:
        health = client.health()
        if health["source_sha256"] != SERVICE_SHA256:
            raise ValueError("live service differs from qualified source")
        for record, solution in selected[:8]:
            verifier = record["metadata"]["verifiers"][0]
            mode = "stdio" if verifier["name"] == "code_stdio" else "assert"
            positive = client.execute(mode, solution, verifier["target"], per_test_seconds=1.0)
            negative = client.execute(mode, "raise RuntimeError('negative control')", verifier["target"])
            if negative.pass_fraction != 0.0:
                raise ValueError("incorrect program received positive credit")
            controls.append(
                {
                    "source_id": record["metadata"]["source_id"],
                    "tests": len(verifier["target"]),
                    "reference_pass_fraction": positive.pass_fraction,
                    "negative_pass_fraction": negative.pass_fraction,
                    "request_ids": positive.request_ids + negative.request_ids,
                }
            )
        if not any(record["reference_pass_fraction"] == 1.0 for record in controls):
            raise ValueError("no fully passing reference control; investigate before training")
    finally:
        client.session.close()
    for name, rows in (("train", records[: args.train_count]), ("eval", records[args.train_count :])):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    write_json(args.output / "verifiers.json", registry)
    report = {
        "dataset": DATASET,
        "revision": REVISION,
        "seed": args.seed,
        "scanned": len(source),
        "eligible": len(candidates),
        "exclusions": exclusions,
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "service": health,
        "controls": controls,
        "selected_ids": [record["metadata"]["prepared_sample_id"] for record in records],
        "total_tests": sum(len(record["metadata"]["verifiers"][0]["target"]) for record in records),
        "sha256": {
            name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
            for name in ("train.jsonl", "eval.jsonl", "verifiers.json")
        },
        "scope": "Deterministic sample from the first 512 source rows; no test-suite truncation or solution leakage.",
    }
    write_json(args.output / "qualification.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
