#!/usr/bin/env python

"""Evaluate a causal LM on Open Instruct math datasets with repeated vLLM samples."""

import argparse
import json
import math
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import vllm
from datasets import load_dataset
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer
from vllm.inputs import TokensPrompt

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES
from open_instruct.ground_truth_utils import MathVerifier

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model ID or local checkpoint path.")
    parser.add_argument("--model-label", required=True, help="Short label recorded in the output files.")
    parser.add_argument("--datasets", nargs="+", required=True, help="Hugging Face math datasets to evaluate.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--chat-template", default="qwen_instruct_user_boxed_math")
    parser.add_argument("--samples-per-prompt", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-response-tokens", type=int, default=16384)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--scoring-workers", type=int, default=16)
    parser.add_argument(
        "--strip-weight-prefix",
        help="Strip this prefix from local safetensors keys into a temporary vLLM-compatible checkpoint.",
    )
    parser.add_argument(
        "--weight-prefix-replacement",
        default="",
        help="Prefix tensor names changed by --strip-weight-prefix with this value.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("/output"))
    return parser.parse_args()


def build_prompts(tokenizer: Any, rows: list[dict[str, Any]], max_prompt_tokens: int) -> list[list[int]]:
    prompts = [
        tokenizer.apply_chat_template(row["messages"], add_generation_prompt=True, tokenize=True)["input_ids"]
        for row in rows
    ]
    too_long = [(index, len(prompt)) for index, prompt in enumerate(prompts) if len(prompt) > max_prompt_tokens]
    if too_long:
        raise ValueError(f"Found prompts longer than {max_prompt_tokens} tokens: {too_long}")
    return prompts


def score_prediction(item: tuple[str, str, str]) -> float:
    prediction, label, query = item
    return MathVerifier()([], prediction, label, query=query).score


def compute_pass_at_k_metrics(correct_per_prompt: np.ndarray) -> dict[str, float]:
    """Compute pass@1, pass@n, and unbiased power-of-two pass@k metrics."""
    correct = np.asarray(correct_per_prompt, dtype=bool)
    num_samples = int(correct.shape[1])
    correct_counts = correct.sum(axis=1).astype(np.int64)
    metrics = {"pass_at_1": float((correct_counts / num_samples).mean())}
    if num_samples > 1:
        metrics[f"pass_at_{num_samples}"] = float((correct_counts > 0).mean())

    k = 1
    while k <= num_samples:
        estimates = [
            1.0 - math.comb(num_samples - int(count), k) / math.comb(num_samples, k)
            if num_samples - int(count) >= k
            else 1.0
            for count in correct_counts
        ]
        metrics[f"pass_at_{k}_unbiased"] = float(np.mean(estimates))
        k *= 2
    return metrics


def remap_weight_name(name: str, weight_prefix: str, replacement_prefix: str) -> str:
    if not name.startswith(weight_prefix):
        return name
    return replacement_prefix + name[len(weight_prefix) :]


def normalize_checkpoint_for_vllm(model: str, weight_prefix: str | None, replacement_prefix: str) -> str:
    """Copy a local checkpoint while replacing an incompatible tensor-name prefix."""
    if not weight_prefix:
        return model

    model_path = Path(model)
    if not model_path.is_dir():
        raise ValueError("--strip-weight-prefix requires --model to be a local checkpoint directory")
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {model_path}")

    output_path = Path(tempfile.mkdtemp(prefix="math-vllm-checkpoint-"))
    for source_path in model_path.iterdir():
        if (
            source_path.is_file()
            and source_path.suffix != ".safetensors"
            and source_path.name != "model.safetensors.index.json"
        ):
            shutil.copy2(source_path, output_path / source_path.name)

    renamed_tensors = 0
    for source_path in weight_files:
        with safe_open(source_path, framework="pt", device="cpu") as source:
            tensors = {}
            for name in list(source.keys()):
                mapped_name = remap_weight_name(name, weight_prefix, replacement_prefix)
                if mapped_name in tensors:
                    raise ValueError(f"Tensor-name collision after replacing {weight_prefix!r}: {mapped_name}")
                tensors[mapped_name] = source.get_tensor(name)
                renamed_tensors += mapped_name != name
            save_file(tensors, output_path / source_path.name, metadata=source.metadata())

    if renamed_tensors == 0:
        raise ValueError(f"No tensor names started with {weight_prefix!r} in {model_path}")

    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["weight_map"] = {
            remap_weight_name(name, weight_prefix, replacement_prefix): shard
            for name, shard in index["weight_map"].items()
        }
        (output_path / index_path.name).write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    logger.info(
        f"Replaced {weight_prefix!r} with {replacement_prefix!r} in {renamed_tensors} tensors into {output_path}"
    )
    return str(output_path)


def summarize(correct_per_prompt: np.ndarray, response_lengths: np.ndarray, finish_reasons: list[str]) -> dict:
    metrics = compute_pass_at_k_metrics(correct_per_prompt)
    metrics.update(
        {
            "num_prompts": int(correct_per_prompt.shape[0]),
            "samples_per_prompt": int(correct_per_prompt.shape[1]),
            "correct_samples": int(correct_per_prompt.sum()),
            "solved_prompts": int(correct_per_prompt.any(axis=1).sum()),
            "mean_response_tokens": float(response_lengths.mean()),
            "max_response_tokens": int(response_lengths.max()),
            "stop_rate": float(np.mean([reason == "stop" for reason in finish_reasons])),
            "truncation_rate": float(np.mean([reason == "length" for reason in finish_reasons])),
        }
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.samples_per_prompt < 1:
        raise ValueError("--samples-per-prompt must be at least 1")
    if args.chat_template not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template: {args.chat_template}")
    if args.model.startswith("/weka/") and not Path(args.model).is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template]

    rows: list[dict[str, Any]] = []
    dataset_ranges: dict[str, tuple[int, int]] = {}
    for dataset_name in args.datasets:
        dataset = load_dataset(dataset_name, split=args.split)
        start = len(rows)
        rows.extend(dict(row) for row in dataset)
        dataset_ranges[dataset_name] = (start, len(rows))
        logger.info(f"Loaded {len(dataset)} prompts from {dataset_name}:{args.split}")

    prompts = build_prompts(tokenizer, rows, args.max_prompt_tokens)
    runtime_model = normalize_checkpoint_for_vllm(args.model, args.strip_weight_prefix, args.weight_prefix_replacement)
    max_model_len = args.max_prompt_tokens + args.max_response_tokens
    llm = vllm.LLM(
        model=runtime_model,
        tokenizer=args.model,
        skip_tokenizer_init=True,
        language_model_only=True,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
        generation_config="vllm",
        disable_cascade_attn=True,
        mamba_ssm_cache_dtype="float32",
        gdn_prefill_backend="triton",
        seed=args.seed,
    )

    responses: list[list[str]] = [[] for _ in rows]
    response_token_ids: list[list[list[int]]] = [[] for _ in rows]
    finish_reasons: list[list[str]] = [[] for _ in rows]
    prompt_payloads = [TokensPrompt(prompt_token_ids=prompt) for prompt in prompts]
    for sample_index in range(args.samples_per_prompt):
        sampling_params = vllm.SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_response_tokens,
            n=1,
            seed=args.seed + sample_index,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        )
        outputs = llm.generate(prompt_payloads, sampling_params=sampling_params)
        for row_index, output in enumerate(outputs):
            completion = output.outputs[0]
            completion_token_ids = list(completion.token_ids)
            responses[row_index].append(tokenizer.decode(completion_token_ids, skip_special_tokens=False))
            response_token_ids[row_index].append(completion_token_ids)
            finish_reasons[row_index].append(completion.finish_reason)

    score_inputs = [
        (responses[row_index][sample_index], str(row["ground_truth"]), row["messages"][0]["content"])
        for row_index, row in enumerate(rows)
        for sample_index in range(args.samples_per_prompt)
    ]
    with ThreadPoolExecutor(max_workers=args.scoring_workers) as executor:
        flat_scores = list(executor.map(score_prediction, score_inputs))
    scores = np.asarray(flat_scores, dtype=np.float64).reshape(len(rows), args.samples_per_prompt)

    sample_path = args.output_dir / "samples.jsonl"
    with sample_path.open("w", encoding="utf-8") as sample_file:
        for row_index, row in enumerate(rows):
            for sample_index in range(args.samples_per_prompt):
                sample_file.write(
                    json.dumps(
                        {
                            "model_label": args.model_label,
                            "model": args.model,
                            "dataset": row["dataset"],
                            "source_dataset": row.get("source_dataset"),
                            "problem_idx": row.get("problem_idx"),
                            "prompt": row["messages"][0]["content"],
                            "ground_truth": row["ground_truth"],
                            "sample_index": sample_index,
                            "response": responses[row_index][sample_index],
                            "response_tokens": len(response_token_ids[row_index][sample_index]),
                            "finish_reason": finish_reasons[row_index][sample_index],
                            "correct": bool(scores[row_index, sample_index]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    summaries: dict[str, Any] = {}
    for dataset_name, (start, end) in dataset_ranges.items():
        dataset_lengths = np.asarray(
            [len(response) for prompt_responses in response_token_ids[start:end] for response in prompt_responses]
        )
        dataset_finish_reasons = [reason for reasons in finish_reasons[start:end] for reason in reasons]
        summaries[dataset_name] = summarize(scores[start:end] > 0.5, dataset_lengths, dataset_finish_reasons)

    all_lengths = np.asarray(
        [len(response) for prompt_responses in response_token_ids for response in prompt_responses]
    )
    all_finish_reasons = [reason for reasons in finish_reasons for reason in reasons]
    summaries["overall"] = summarize(scores > 0.5, all_lengths, all_finish_reasons)

    summary = {
        "model_label": args.model_label,
        "model": args.model,
        "split": args.split,
        "chat_template": args.chat_template,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_response_tokens": args.max_response_tokens,
        "seed": args.seed,
        "stripped_weight_prefix": args.strip_weight_prefix,
        "weight_prefix_replacement": args.weight_prefix_replacement,
        "datasets": summaries,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info(f"Wrote samples to {sample_path}")
    logger.info(f"Wrote summary to {summary_path}")
    logger.info(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
