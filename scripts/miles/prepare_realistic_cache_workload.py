"""Freeze the published Dolci RL prompt mixture for reward-free cache measurements.

This is a prompt-only benchmark artifact, not a training-data conversion. Source
answers/validators are retained as provenance; invoking its reward factory fails.
"""

import argparse
import collections
import hashlib
import json
import re
import shutil
from pathlib import Path

import datasets
import numpy as np

from open_instruct.miles.datasets import run_data

SOURCE = "allenai/Dolci-Think-RL-7B"
REVISION = "0fb6466d31ef3a9dd16985ef635e6429e05a6491"
COLUMNS = [
    "prompt",
    "input_ids_prompt",
    "dataset_source",
    "original_dataset",
    "custom_id",
    "id",
    "conversation_hash",
    "dataset",
    "ground_truth",
]
MESSAGE = re.compile(r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>\n", re.DOTALL)
GENERATION_SUFFIX = "<|im_start|>assistant\n<think>"


def no_reward(*args, **kwargs):
    raise RuntimeError("Prompt-only cache workload requires benchmark.reward_mode=none; no training rewards exist")


def reconstruct_messages(raw_prompt, decoded):
    """Use tokenized boundaries only when contents exactly match the source text."""
    messages, cursor = [], 0
    for match in MESSAGE.finditer(decoded):
        if match.start() != cursor:
            raise ValueError("Non-contiguous source ChatML messages")
        messages.append({"role": match[1], "content": match[2]})
        cursor = match.end()
    if decoded[cursor:] not in (GENERATION_SUFFIX, GENERATION_SUFFIX + "\n"):
        raise ValueError("Unexpected source generation suffix or unparsed message")
    for candidate in (messages, messages[1:] if messages and messages[0]["role"] == "system" else []):
        rendered = "\n".join(f"{message['role']}: {message['content']}" for message in candidate)
        if candidate and rendered == raw_prompt and candidate[-1]["role"] == "user":
            return candidate
    raise ValueError("Source text does not exactly match decoded message contents; refusing lossy conversion")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def prepare(model, output, seed=17):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, output / "prepare_realistic_cache_workload.py")
    tokenizer = run_data._tokenizer(model)
    tokenizer.save_pretrained(output / "tokenizer")
    source = datasets.load_dataset(SOURCE, revision=REVISION, split="train", columns=COLUMNS)
    if len(source) != 102014:
        raise ValueError(f"Pinned source row count changed: {len(source)}")
    rows, inventory, failures = [], [], []
    with (output / "source-records.jsonl").open("w") as raw_file:
        for index, row in enumerate(source):
            raw_file.write(json.dumps({"source_row": index, **row}, ensure_ascii=False) + "\n")
            try:
                decoded = tokenizer.decode(row["input_ids_prompt"], skip_special_tokens=False)
                messages = reconstruct_messages(row["prompt"], decoded)
                prompt = run_data._render(messages, tokenizer, tokenizer.chat_template)
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                identity = f"{SOURCE}@{REVISION}:train:{index}"
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                record = {
                    "prepared_sample_id": identity,
                    "source_row": index,
                    "dataset_source": row["dataset_source"],
                    "original_dataset": row["original_dataset"],
                    "custom_id": row["custom_id"],
                    "prompt_sha256": digest,
                    "prompt_tokens": len(ids),
                }
                inventory.append(record)
                rows.append(
                    {
                        "input": prompt,
                        "label": None,
                        "metadata": {
                            **record,
                            "benchmark_prompt_only": True,
                            "verifiers": [{"name": "prompt_only_requires_reward_mode_none", "target": None}],
                        },
                    }
                )
            except (ValueError, TypeError, KeyError) as error:
                failures.append({"source_row": index, "custom_id": row.get("custom_id"), "error": str(error)})
            if index % 5000 == 0:
                print(json.dumps({"processed": index + 1, "failures": len(failures)}), flush=True)
    write_json(output / "conversion-failures.json", failures)
    if failures:
        raise ValueError(f"{len(failures)} source rows need explicit reconstruction; no filtered artifact published")
    rows.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['metadata']['prepared_sample_id']}".encode()).hexdigest())
    inventory_by_id = {row["prepared_sample_id"]: row for row in inventory}
    with (output / "order.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            stream.write(
                json.dumps({"stream_index": index, **inventory_by_id[row["metadata"]["prepared_sample_id"]]}) + "\n"
            )
    lengths = [row["prompt_tokens"] for row in inventory]
    counts = collections.Counter(row["dataset_source"] for row in inventory)
    duplicates = collections.Counter(row["prompt_sha256"] for row in inventory)
    summary = {
        "source": SOURCE,
        "revision": REVISION,
        "rows": len(rows),
        "seed": seed,
        "source_counts": dict(counts),
        "filtered_or_truncated_rows": 0,
        "prompt_tokens": dict(
            zip(
                ["min", "p50", "p90", "p95", "p99", "max"],
                np.quantile(lengths, [0, 0.5, 0.9, 0.95, 0.99, 1]).tolist(),
                strict=True,
            )
        ),
        "unique_rendered_prompts": len(duplicates),
        "duplicate_rows": sum(n - 1 for n in duplicates.values()),
        "scope": "Published Olmo 3 RL mixture proxy; not an established Olmo 3.5 production distribution. Reward-free benchmark only.",
        "ordering": "Frozen SHA256(seed:prepared_sample_id) order; set data.shuffle=false. group_index maps to stream_index before first epoch rollover.",
        "conversion": "Source ChatML boundaries decoded with target tokenizer; message contents exactly verified against source prompt text. Implicit source system removed; explicit messages rerendered with target checkpoint chat template.",
        "reward_contract": "benchmark.reward_mode=none required; original ground_truth/dataset preserved only in source-records.jsonl.",
    }
    write_json(output / "summary.json", summary)
    source_dir = output / "source"
    source_dir.mkdir()
    with (source_dir / "train.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_json(
        source_dir / "verifiers.json",
        {
            "prompt_only_requires_reward_mode_none": {
                "factory": "scripts.miles.prepare_realistic_cache_workload.no_reward"
            }
        },
    )
    del rows, source, inventory_by_id
    result = run_data.prepare_data(
        {"prompt_data": str(source_dir / "train.jsonl"), "reward_config": str(source_dir / "verifiers.json")},
        Path(model),
        output / "prepared",
        max_prompt_length=max(lengths),
        seed=seed,
    )
    write_json(output / "result.json", result)
    manifest = []
    for path in sorted(output.rglob("*")):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            manifest.append({"path": str(path.relative_to(output)), "bytes": path.stat().st_size, "sha256": digest})
    write_json(output / "hashes.json", manifest)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    prepare(args.model, args.output, args.seed)


if __name__ == "__main__":
    main()
