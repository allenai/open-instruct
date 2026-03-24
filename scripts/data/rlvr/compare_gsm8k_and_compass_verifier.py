#!/usr/bin/env python3
"""
Annotate a GSM8K completion dataset with both GSM8K and CompassVerifier scores.

This script expects a dataset with one row per prompt and a `completions` column
containing a list of candidate outputs. It produces:
1. A prompt-level dataset with per-completion score arrays and pass-rate summaries.
2. A disagreement-only JSONL file with one row per mismatched completion.
3. A summary JSON file with aggregate comparison stats.
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct import logger_utils
from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.judge_utils import compass_verifier_template, extract_score_compass_verifier
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)

COMPASS_PLACEHOLDER = "__COMPASS_LLM_RESPONSE_PLACEHOLDER__"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GSM8K and CompassVerifier on a completion dataset.")
    parser.add_argument(
        "--dataset",
        default="mnoukhov/gsm8k-platinum-openinstruct-qwen2.5-0.5b-instruct-1024samples-buckets",
        help="Input HF dataset",
    )
    parser.add_argument("--split", default="test", help="Input split")
    parser.add_argument(
        "--judge-model",
        default="opencompass/CompassVerifier-3B",
        help="CompassVerifier model to use for judging",
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85, help="vLLM GPU memory utilization")
    parser.add_argument("--max-model-len", type=int, default=2048, help="Max model length for CompassVerifier")
    parser.add_argument("--judge-max-tokens", type=int, default=32, help="Max judge output tokens")
    parser.add_argument("--batch-size", type=int, default=256, help="Number of completions to judge per batch")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for vLLM")
    parser.add_argument("--limit-rows", type=int, default=None, help="Optional number of dataset rows to process")
    parser.add_argument(
        "--limit-completions-per-prompt",
        type=int,
        default=None,
        help="Optional number of completions to score per prompt",
    )
    parser.add_argument(
        "--save-local-dir",
        default="/tmp/gsm8k_compass_verifier_comparison",
        help="Base output directory for local artifacts",
    )
    return parser.parse_args()


def get_query(sample: dict) -> str:
    if "prompt" in sample and sample["prompt"]:
        return sample["prompt"]
    if "messages" in sample and sample["messages"]:
        return "\n".join(msg["content"] for msg in sample["messages"] if msg["role"] == "user")
    raise ValueError("Sample is missing both `prompt` and `messages` content.")


def build_compass_chat_prompt(
    tokenizer: AutoTokenizer,
    question: str,
    gold_answer: str,
    completion: str,
    max_model_len: int,
    judge_max_tokens: int,
) -> str:
    prompt = compass_verifier_template.format(
        question=question,
        gold_answer=gold_answer,
        llm_response=COMPASS_PLACEHOLDER,
    )
    prefix, suffix = prompt.split(COMPASS_PLACEHOLDER)

    # Reserve output tokens and a small buffer for chat-template overhead.
    max_input_tokens = max_model_len - judge_max_tokens - 16
    if max_input_tokens < 128:
        raise ValueError(f"max_model_len={max_model_len} leaves too little room for CompassVerifier input.")

    base_messages = [{"role": "user", "content": prefix + suffix}]
    base_token_ids = tokenizer.apply_chat_template(base_messages, add_generation_prompt=True, tokenize=True)
    response_budget = max(1, max_input_tokens - len(base_token_ids))

    response_token_ids = tokenizer.encode(completion, add_special_tokens=False)
    trimmed_response = tokenizer.decode(response_token_ids[-response_budget:])
    final_messages = [{"role": "user", "content": prefix + trimmed_response + suffix}]
    return tokenizer.apply_chat_template(final_messages, add_generation_prompt=True, tokenize=False)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    args = parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.save_local_dir) / f"{args.split}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s[%s]", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, num_proc=max_num_processes())
    if args.limit_rows is not None:
        ds = ds.select(range(min(args.limit_rows, len(ds))))
    logger.info("Loaded %d prompts", len(ds))

    tokenizer = AutoTokenizer.from_pretrained(args.judge_model)
    llm = LLM(
        model=args.judge_model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.judge_max_tokens,
        seed=args.seed,
    )

    gsm8k_verifier = GSM8KVerifier()
    prompt_rows = []
    disagreement_rows = []
    aggregate = {
        "num_prompts": len(ds),
        "total_completions": 0,
        "gsm8k_pass_count": 0,
        "compass_pass_count": 0,
        "both_pass": 0,
        "both_fail": 0,
        "only_gsm8k_pass": 0,
        "only_compass_pass": 0,
    }

    pending_prompts: list[str] = []
    pending_meta: list[tuple[int, int]] = []
    row_results: dict[int, dict[str, list[int]]] = {}

    def flush_pending() -> None:
        if not pending_prompts:
            return
        outputs = llm.generate(pending_prompts, sampling_params)
        for request_output, (row_idx, completion_idx) in zip(outputs, pending_meta):
            judgment = request_output.outputs[0].text
            score = int(extract_score_compass_verifier(judgment)[1])
            row_results[row_idx]["compass_scores"][completion_idx] = score
            row_results[row_idx]["compass_judgments"][completion_idx] = judgment
        pending_prompts.clear()
        pending_meta.clear()

    for row_idx, sample in enumerate(ds):
        query = get_query(sample)
        ground_truth = str(sample["ground_truth"])
        completions = sample["completions"]
        if args.limit_completions_per_prompt is not None:
            completions = completions[: args.limit_completions_per_prompt]
        gsm8k_scores: list[int] = []

        row_results[row_idx] = {
            "compass_scores": [-1] * len(completions),
            "compass_judgments": [""] * len(completions),
        }

        for completion_idx, completion in enumerate(completions):
            gsm8k_score = int(gsm8k_verifier([], completion, ground_truth).score)
            gsm8k_scores.append(gsm8k_score)

            pending_prompts.append(
                build_compass_chat_prompt(
                    tokenizer=tokenizer,
                    question=query,
                    gold_answer=ground_truth,
                    completion=completion,
                    max_model_len=args.max_model_len,
                    judge_max_tokens=args.judge_max_tokens,
                )
            )
            pending_meta.append((row_idx, completion_idx))

            if len(pending_prompts) >= args.batch_size:
                flush_pending()

        flush_pending()

        compass_scores = row_results[row_idx]["compass_scores"]
        compass_judgments = row_results[row_idx]["compass_judgments"]
        disagreements = 0
        both_pass = 0
        both_fail = 0
        only_gsm8k_pass = 0
        only_compass_pass = 0

        for completion_idx, (completion, gsm8k_score, compass_score, compass_judgment) in enumerate(
            zip(completions, gsm8k_scores, compass_scores, compass_judgments)
        ):
            if gsm8k_score and compass_score:
                both_pass += 1
            elif (not gsm8k_score) and (not compass_score):
                both_fail += 1
            elif gsm8k_score and not compass_score:
                only_gsm8k_pass += 1
                disagreements += 1
            else:
                only_compass_pass += 1
                disagreements += 1

            if gsm8k_score != compass_score:
                disagreement_rows.append(
                    {
                        "row_idx": row_idx,
                        "completion_idx": completion_idx,
                        "dataset": sample["dataset"],
                        "prompt": query,
                        "ground_truth": sample["ground_truth"],
                        "completion": completion,
                        "gsm8k_score": gsm8k_score,
                        "compass_score": compass_score,
                        "compass_judgment": compass_judgment,
                    }
                )

        num_completions = len(completions)
        gsm8k_pass_count = sum(gsm8k_scores)
        compass_pass_count = sum(compass_scores)

        aggregate["total_completions"] += num_completions
        aggregate["gsm8k_pass_count"] += gsm8k_pass_count
        aggregate["compass_pass_count"] += compass_pass_count
        aggregate["both_pass"] += both_pass
        aggregate["both_fail"] += both_fail
        aggregate["only_gsm8k_pass"] += only_gsm8k_pass
        aggregate["only_compass_pass"] += only_compass_pass

        prompt_rows.append(
            {
                **sample,
                "gsm8k_scores": gsm8k_scores,
                "compass_scores": compass_scores,
                "compass_judgments": compass_judgments,
                "gsm8k_pass_count_recomputed": gsm8k_pass_count,
                "compass_pass_count": compass_pass_count,
                "gsm8k_pass_rate_recomputed": gsm8k_pass_count / num_completions,
                "compass_pass_rate": compass_pass_count / num_completions,
                "agreement_count": num_completions - disagreements,
                "disagreement_count": disagreements,
                "both_pass_count": both_pass,
                "both_fail_count": both_fail,
                "only_gsm8k_pass_count": only_gsm8k_pass,
                "only_compass_pass_count": only_compass_pass,
            }
        )

        if (row_idx + 1) % 10 == 0 or row_idx + 1 == len(ds):
            logger.info(
                "Processed %d/%d prompts (%d completions)",
                row_idx + 1,
                len(ds),
                aggregate["total_completions"],
            )

    aggregate["gsm8k_pass_rate"] = aggregate["gsm8k_pass_count"] / aggregate["total_completions"]
    aggregate["compass_pass_rate"] = aggregate["compass_pass_count"] / aggregate["total_completions"]
    aggregate["agreement_rate"] = (aggregate["both_pass"] + aggregate["both_fail"]) / aggregate["total_completions"]
    aggregate["disagreement_rate"] = (
        aggregate["only_gsm8k_pass"] + aggregate["only_compass_pass"]
    ) / aggregate["total_completions"]

    prompt_ds = Dataset.from_list(prompt_rows)
    prompt_ds_dir = output_dir / "prompt_level_dataset"
    prompt_ds.save_to_disk(str(prompt_ds_dir))
    prompt_ds.to_json(str(output_dir / "prompt_level.jsonl"))

    disagreement_path = output_dir / "disagreements.jsonl"
    with disagreement_path.open("w") as f:
        for row in disagreement_rows:
            f.write(json.dumps(row) + "\n")

    save_json(output_dir / "summary.json", aggregate)

    logger.info("Saved prompt-level dataset to %s", prompt_ds_dir)
    logger.info("Saved %d disagreement rows to %s", len(disagreement_rows), disagreement_path)
    logger.info("Saved summary to %s", output_dir / "summary.json")
    logger.info(
        "Aggregate pass rates: gsm8k=%.4f compass=%.4f disagreement=%.4f",
        aggregate["gsm8k_pass_rate"],
        aggregate["compass_pass_rate"],
        aggregate["disagreement_rate"],
    )


if __name__ == "__main__":
    main()
