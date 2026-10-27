"""Render full GSM8K/platinum splits for the base MainlineLadder checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import datasets
import transformers
from huggingface_hub import HfApi
from transformers_plugin import register_config

TRAIN = "ai2-adapt-dev/rlvr_gsm8k_zs"
EVAL = "mnoukhov/gsm8k-platinum-openinstruct"
REGISTRY = {"gsm8k": {"factory": "open_instruct.miles.rewards.verifiers.GSM8KVerifier"}}


def render(row):
    messages = row["messages"]
    prompt_messages = messages if len(messages) == 1 else messages[:-1]
    if len(prompt_messages) != 1 or prompt_messages[0]["role"] != "user":
        raise ValueError("Expected one GSM8K user question")
    return f"Question: {prompt_messages[0]['content']}\nAnswer:"


def prepare(dataset, split, tokenizer, output, max_prompt_tokens):
    revision = HfApi().dataset_info(dataset).sha
    records = datasets.load_dataset(dataset, revision=revision, split=split)
    written = dropped = 0
    with output.open("w") as stream:
        for index, row in enumerate(records):
            prompt = render(row)
            if len(tokenizer.encode(prompt, add_special_tokens=False)) > max_prompt_tokens:
                dropped += 1
                continue
            target = row["ground_truth"]
            if isinstance(target, list):
                if len(target) != 1:
                    raise ValueError("Expected one ground truth")
                target = target[0]
            record = {
                "input": prompt,
                "label": target,
                "metadata": {
                    "prepared_sample_id": f"{dataset}:{revision}:{split}:{index}",
                    "query": row["messages"][0]["content"],
                    "verifiers": [{"name": "gsm8k", "target": target, "weight": 1.0}],
                },
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return {
        "dataset": dataset,
        "revision": revision,
        "split": split,
        "written": written,
        "dropped_over_length": dropped,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    args = parser.parse_args()
    register_config()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "tokenizer": args.tokenizer,
        "max_prompt_tokens": args.max_prompt_tokens,
        "prompt_format": "gsm8k_base",
        "train": prepare(TRAIN, "train", tokenizer, args.output / "train.jsonl", args.max_prompt_tokens),
        "eval": prepare(EVAL, "test", tokenizer, args.output / "eval.jsonl", args.max_prompt_tokens),
    }
    (args.output / "verifiers.json").write_text(json.dumps(REGISTRY, indent=2) + "\n")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
