"""Prepare the Qwen3 Deepscaler benchmark as MILES prompt_data JSONL.

Mirrors the open-instruct OLMo-core baseline ai2-llm/open_instruct_internal/cmoi4l53
(qwen3_4b_base_deepscaler_oc_2k_baseline_dapo_n8_k16): train on
mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles and evaluate on
allenai/aime_2025_openinstruct, rendered with the qwen_instruct_user_boxed_math chat template
and no system prompt, as rlvr_tokenize_v1 does. Rewards use the MILES-native 1/0 scale
(open-instruct's verification_reward is 10). Writes train.jsonl, one JSONL per eval set and
verifiers.json into OUTPUT, for [data] prompt_data / eval_prompt_data / reward_config.

Prompts longer than --max-prompt-tokens are dropped, matching open-instruct's
rlvr_max_length_filter_v1; MILES rejects over-long prompts instead of skipping them.

Usage: python scripts/data/miles/prepare_deepscaler.py /weka/.../data/deepscaler-10k-base-01 --tokenizer /weka/.../Qwen3-4B-Base
"""

import argparse
import json
from pathlib import Path

import datasets
import transformers

from open_instruct import dataset_transformation

TRAIN = "mnoukhov/deepscaler-10k-qwen3-4b-base-32samples-quartiles"
EVALS = {"aime_2025": "allenai/aime_2025_openinstruct"}
CHAT_TEMPLATE = "qwen_instruct_user_boxed_math"
VERIFICATION_REWARD = 1.0
REGISTRY = {"math": {"factory": "open_instruct.ground_truth_utils.MathVerifier"}}


def rows(name, dataset, tokenizer, max_prompt_tokens):
    dropped = 0
    for index, row in enumerate(datasets.load_dataset(dataset, split="train")):
        # rlvr_tokenize_v1: a single message is the prompt, otherwise all but the last turn.
        messages = row["messages"] if len(row["messages"]) == 1 else row["messages"][:-1]
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
                "verifiers": [{"name": "math", "target": target, "weight": VERIFICATION_REWARD}],
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
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-4B-Base")
    parser.add_argument("--train-dataset", default=TRAIN)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    options = parser.parse_args()
    tokenizer = transformers.AutoTokenizer.from_pretrained(options.tokenizer)
    tokenizer.chat_template = dataset_transformation.CHAT_TEMPLATES[CHAT_TEMPLATE]
    options.output.mkdir(parents=True, exist_ok=False)
    write(
        options.output / "train.jsonl", rows("deepscaler", options.train_dataset, tokenizer, options.max_prompt_tokens)
    )
    for name, dataset in EVALS.items():
        write(options.output / f"eval-{name}.jsonl", rows(name, dataset, tokenizer, options.max_prompt_tokens))
    (options.output / "verifiers.json").write_text(json.dumps(REGISTRY, indent=2) + "\n")


if __name__ == "__main__":
    main()
