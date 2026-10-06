"""Prepare the never-give-up Deepscaler experiment as MILES prompt_data JSONL.

Mirrors mnoukhov/never-give-up experiments/deepscaler/qwen3_4b_deepscaler.sh: train on
mnoukhov/deepscaler_openinstruct and evaluate on AIME/BRUMO/HMMT 2025, with the math system
prompt prepended and the policy's own chat template. Writes train.jsonl, one JSONL per eval set
and verifiers.json into OUTPUT, for [data] prompt_data / eval_prompt_data / reward_config.

Prompts longer than --max-prompt-tokens are dropped, matching open-instruct's --max_prompt_token_length
filter; MILES rejects over-long prompts instead of skipping them.

Usage: python scripts/miles/prepare_deepscaler.py /weka/.../data/deepscaler-10k --train-dataset mnoukhov/deepscaler-10k-rlvr
"""

import argparse
import json
from pathlib import Path

import datasets
import transformers

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
TRAIN = "mnoukhov/deepscaler_openinstruct"
EVALS = {
    "aime_2025": "mnoukhov/aime_2025_openinstruct",
    "brumo_2025": "mnoukhov/brumo_2025_openinstruct",
    "hmmt_nov_2025": "mnoukhov/hmmt_nov_2025_openinstruct",
    "hmmt_feb_2025": "mnoukhov/hmmt_feb_2025_openinstruct",
}
REGISTRY = {"math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"}}


def rows(name, dataset, tokenizer, max_prompt_tokens):
    dropped = 0
    for index, row in enumerate(datasets.load_dataset(dataset, split="train")):
        messages = [message for message in row["messages"] if message["role"] != "assistant"]
        # open-instruct's --system_prompt_override_file replaces any system turn.
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [m for m in messages if m["role"] != "system"]
        target = row["ground_truth"]
        if isinstance(target, list):
            if len(target) != 1:
                raise ValueError(f"{dataset} row {index}: expected one ground truth, got {len(target)}")
            target = target[0]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(prompt, add_special_tokens=False).input_ids) > max_prompt_tokens:
            dropped += 1
            continue
        yield {
            "input": prompt,
            "label": target,
            "metadata": {
                "prepared_sample_id": f"{name}:{index}",
                "source_dataset": dataset,
                "source_row": index,
                "query": messages[-1]["content"],
                "verifiers": [{"name": "math", "target": target, "weight": 1.0}],
            },
        }
    print(f"Dropped {dropped} {dataset} prompts over {max_prompt_tokens} tokens")


def write(path, records):
    with path.open("w") as stream:
        count = 0
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print(f"Wrote {count} rows to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Thinking-2507")
    parser.add_argument("--train-dataset", default=TRAIN)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    options = parser.parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.tokenizer)
    options.output.mkdir(parents=True, exist_ok=False)
    write(options.output / "train.jsonl", rows("deepscaler", options.train_dataset, tokenizer, options.max_prompt_tokens))
    for name, dataset in EVALS.items():
        write(options.output / f"eval-{name}.jsonl", rows(name, dataset, tokenizer, options.max_prompt_tokens))
    (options.output / "verifiers.json").write_text(json.dumps(REGISTRY, indent=2) + "\n")


if __name__ == "__main__":
    main()
