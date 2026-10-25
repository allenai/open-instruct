"""Prepare a pinned, explicitly length-filtered mixed inference benchmark corpus."""

import argparse
import hashlib
import json
from pathlib import Path

from open_instruct.miles.datasets import run_data


def prepare(model, output, *, counts, seed=17, max_prompt_length=2048):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tokenizer = run_data._tokenizer(model)
    selected = []
    selection = []
    for name, count in counts.items():
        tasks = {"tasks": [{"task": name, "train_count": 2 * count}], "seed": seed, "shuffle": False}
        partitions, provenance, _ = run_data._tasks(tasks, tokenizer, tokenizer.chat_template, seed)
        accepted, rejected = [], []
        for row in partitions["train"]:
            length = len(tokenizer.encode(row["input"], add_special_tokens=False))
            if length > max_prompt_length:
                rejected.append({"id": row["metadata"]["prepared_sample_id"], "prompt_tokens": length})
            elif len(accepted) < count:
                accepted.append(row)
        if len(accepted) != count:
            raise ValueError(f"Not enough {name} prompts fit the explicit {max_prompt_length}-token limit")
        selected.extend(accepted)
        selection.append(
            {
                "task": name,
                "provenance": provenance,
                "selected": [r["metadata"]["prepared_sample_id"] for r in accepted],
                "rejected_overlength": rejected,
            }
        )
    # This deterministic permutation is part of the immutable artifact.
    selected.sort(
        key=lambda row: hashlib.sha256(f"{seed}:{row['metadata']['prepared_sample_id']}".encode()).hexdigest()
    )
    source = output / "source"
    source.mkdir()
    (source / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected))
    names = {spec["name"] for row in selected for spec in row["metadata"]["verifiers"]}
    (source / "verifiers.json").write_text(
        json.dumps({name: {"factory": run_data.FACTORIES[name]} for name in sorted(names)})
    )
    (output / "selection.json").write_text(
        json.dumps({"seed": seed, "max_prompt_length": max_prompt_length, "selection": selection}, indent=2) + "\n"
    )
    result = run_data.prepare_data(
        {"prompt_data": str(source / "train.jsonl"), "reward_config": str(source / "verifiers.json")},
        Path(model),
        output / "prepared",
        max_prompt_length=max_prompt_length,
        seed=seed,
    )
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", action="append", required=True, help="Pinned task=count, e.g. math=1024")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    args = parser.parse_args()
    counts = {}
    for entry in args.task:
        name, _, count = entry.partition("=")
        if name not in run_data.TASKS or name in counts or not count.isdecimal() or int(count) <= 0:
            parser.error("Each task must be a distinct supported name with a positive count")
        counts[name] = int(count)
    prepare(args.model, args.output, counts=counts, seed=args.seed, max_prompt_length=args.max_prompt_length)


if __name__ == "__main__":
    main()
