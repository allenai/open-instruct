"""Generate pass@k completions for Manufactoria and attach per-test pass statistics.

Example:
uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \
  --dataset manufactoria/basic_mix_test \
  --split train \
  --model Qwen/Qwen3-0.6B \
  --chat-template tulu \
  --num-samples 32 \
  --num-engines 4 \
  --push-to-hub <entity>/manufactoria-basic-mix-test-pass32
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import ray
from datasets import Dataset, load_dataset
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct import logger_utils
from open_instruct.code_utils.manufactoria_parser import ParseError, create_robot_factory
from open_instruct.dataset_transformation import CHAT_TEMPLATES
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pass@k completions for Manufactoria datasets.")
    parser.add_argument("--dataset", default="manufactoria/basic_mix_test", help="Input HF dataset")
    parser.add_argument("--split", default="train", help="Input split")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B", help="Model used for generation")
    parser.add_argument("--chat-template", default="tulu", help="Chat template name")
    parser.add_argument("--num-samples", type=int, default=32, help="Completions per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per completion")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument("--num-engines", "--num_engines", dest="num_engines", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process")
    parser.add_argument(
        "--save-local-dir",
        default="/tmp/manufactoria_pass_at_k_dataset_outputs",
        help="Base directory for saving generated dataset artifacts before push",
    )
    parser.add_argument("--push-to-hub", default="", help="Output HF dataset repo; set empty string to skip push")
    parser.add_argument("--private", action="store_true", help="Push dataset as private (default public)")
    return parser.parse_args()


def extract_generation_messages(sample: dict[str, Any]) -> list[dict[str, str]]:
    prompt = sample.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return [{"role": "user", "content": prompt.strip()}]

    messages = sample.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return [{"role": "user", "content": content.strip()}]

    raise ValueError("Sample is missing a usable prompt or user message")


def build_prompts(ds: Dataset, model_name: str, chat_template_name: str) -> tuple[list[str], list[list[dict[str, str]]]]:
    if chat_template_name not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template: {chat_template_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.chat_template = CHAT_TEMPLATES[chat_template_name]

    generation_messages = [extract_generation_messages(sample) for sample in ds]
    prompts = [
        tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in generation_messages
    ]
    return prompts, generation_messages


def _split_evenly(items: list[str], num_chunks: int) -> list[tuple[int, list[str]]]:
    base, extra = divmod(len(items), num_chunks)
    chunks = []
    start = 0
    for chunk_idx in range(num_chunks):
        chunk_size = base + (1 if chunk_idx < extra else 0)
        end = start + chunk_size
        chunks.append((start, items[start:end]))
        start = end
    return chunks


@ray.remote
def _generate_completions_chunk(
    prompt_offset: int,
    prompts: list[str],
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    sampling_params_kwargs: dict[str, Any],
) -> tuple[int, list[list[str]]]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="uni" if tensor_parallel_size == 1 else "mp",
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    outputs = llm.generate(prompts, SamplingParams(**sampling_params_kwargs))
    completions = [[completion.text for completion in request_output.outputs] for request_output in outputs]
    return prompt_offset, completions


def generate_completions(
    prompts: list[str],
    model: str,
    num_engines: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    sampling_params: SamplingParams,
) -> list[list[str]]:
    if num_engines < 1:
        raise ValueError(f"--num-engines must be >= 1, got {num_engines}")

    started_ray = not ray.is_initialized()
    if started_ray:
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": dict(os.environ)})

    pg = placement_group(
        [{"GPU": tensor_parallel_size, "CPU": tensor_parallel_size} for _ in range(num_engines)], strategy="PACK"
    )
    ray.get(pg.ready())

    try:
        chunks = [(offset, chunk) for offset, chunk in _split_evenly(prompts, num_engines) if chunk]
        sampling_params_kwargs = dict(
            n=sampling_params.n,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=sampling_params.max_tokens,
            seed=sampling_params.seed,
        )
        futures = []
        for bundle_index, (offset, chunk) in enumerate(chunks):
            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_index,
            )
            futures.append(
                _generate_completions_chunk.options(
                    num_cpus=tensor_parallel_size,
                    num_gpus=tensor_parallel_size,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    prompt_offset=offset,
                    prompts=chunk,
                    model=model,
                    tensor_parallel_size=tensor_parallel_size,
                    gpu_memory_utilization=gpu_memory_utilization,
                    sampling_params_kwargs=sampling_params_kwargs,
                )
            )

        completions_by_prompt: list[list[str] | None] = [None] * len(prompts)
        for offset, chunk_completions in ray.get(futures):
            for idx, completion_list in enumerate(chunk_completions):
                completions_by_prompt[offset + idx] = completion_list

        if any(completions is None for completions in completions_by_prompt):
            raise RuntimeError("Some prompt completions were not returned by the multi-engine generation path.")

        return completions_by_prompt
    finally:
        ray.util.remove_placement_group(pg)
        if started_ray:
            ray.shutdown()


def normalize_test_cases(ground_truth: Any) -> list[dict[str, Any]]:
    if isinstance(ground_truth, str):
        test_cases = json.loads(ground_truth)
    else:
        test_cases = ground_truth

    if not isinstance(test_cases, list) or not test_cases:
        raise ValueError("Manufactoria ground truth must be a non-empty list of test cases")
    return [dict(test_case) for test_case in test_cases]


def extract_manufactoria_code(model_output: str) -> str:
    pattern = r"```(?:manufactoria)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)
    if not matches:
        return model_output
    return matches[-1].strip()


def run_test_case(factory, test_case: dict[str, Any]) -> bool:
    result = factory.process_robot(str(test_case.get("input", "")))
    check_output = bool(test_case.get("check_output", True))
    expected_output = str(test_case.get("expected_output", ""))
    expected_accepted = bool(test_case.get("expected_accepted", True))

    if check_output:
        has_regex_patterns = any(char in expected_output for char in [".", "+", "*", "?", "|", "(", ")"])
        if has_regex_patterns:
            try:
                output_matches = bool(re.fullmatch(expected_output, result.final_tape))
            except re.error:
                output_matches = result.final_tape == expected_output
        else:
            output_matches = result.final_tape == expected_output
        return (output_matches and result.finished) == expected_accepted

    return result.finished == expected_accepted


def score_completion(completion: str, test_cases: list[dict[str, Any]]) -> tuple[bool, list[bool]]:
    try:
        factory = create_robot_factory(extract_manufactoria_code(completion))
    except (ParseError, ValueError):
        return False, [False] * len(test_cases)

    per_test_passes = [run_test_case(factory, test_case) for test_case in test_cases]
    return all(per_test_passes), per_test_passes


def build_output_rows(
    ds: Dataset,
    generation_messages: list[list[dict[str, str]]],
    completions_by_prompt: list[list[str]],
    args: argparse.Namespace,
    split: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    logger.info("Scoring completions and building output rows for %s", split)
    for sample, sanitized_messages, completions in zip(ds, generation_messages, completions_by_prompt):
        test_cases = normalize_test_cases(sample["ground_truth"])
        test_pass_counts = [0] * len(test_cases)
        all_pass_count = 0

        for completion in completions:
            all_passed, per_test_passes = score_completion(completion, test_cases)
            all_pass_count += int(all_passed)
            for test_index, passed in enumerate(per_test_passes):
                test_pass_counts[test_index] += int(passed)

        row = dict(sample)
        row["messages"] = sanitized_messages
        row["completions"] = completions
        row["pass_count"] = all_pass_count
        row["pass_rate"] = f"{all_pass_count}/{args.num_samples}"
        row["all_pass_count"] = all_pass_count
        row["all_pass_rate"] = all_pass_count / args.num_samples
        row["test_pass_counts"] = test_pass_counts
        row["test_pass_rates"] = [count / args.num_samples for count in test_pass_counts]
        row["num_samples"] = args.num_samples
        row["generator_model"] = args.model
        row["generator_chat_template"] = args.chat_template
        row["generator_temperature"] = args.temperature
        row["generator_top_p"] = args.top_p
        row["generator_max_tokens"] = args.max_tokens
        row["source_split"] = split
        rows.append(row)

    flattened_tests: list[tuple[int, int, float]] = []
    next_test_global_index = 0
    dataset_stub = args.dataset.split("/")[-1].replace("-", "_")

    for row_index, row in enumerate(rows):
        updated_test_cases = normalize_test_cases(row["ground_truth"])
        updated_test_summaries: list[dict[str, Any]] = []
        for test_index, (test_case, pass_count) in enumerate(zip(updated_test_cases, row["test_pass_counts"])):
            pass_rate = pass_count / args.num_samples
            test_case["test_index"] = test_index
            test_case["test_global_index"] = next_test_global_index
            test_case.setdefault("test_id", f"{split}_{next_test_global_index}")
            updated_test_summaries.append(
                {
                    "test_index": test_index,
                    "test_global_index": next_test_global_index,
                    "test_id": test_case["test_id"],
                    "description": test_case.get("description", ""),
                    "pass_count": pass_count,
                    "pass_rate": pass_rate,
                }
            )
            flattened_tests.append((row_index, test_index, pass_rate))
            next_test_global_index += 1
        row["ground_truth"] = json.dumps(updated_test_cases)
        row["ground_truth_tests"] = updated_test_cases
        row["test_summaries"] = updated_test_summaries

    sorted_test_positions = [position for position, _ in sorted(enumerate(flattened_tests), key=lambda item: item[1][2])]
    quartile_members = [set(chunk.tolist()) for chunk in np.array_split(sorted_test_positions, 4)]

    for quartile, member_positions in enumerate(quartile_members):
        for flattened_position in member_positions:
            row_index, test_index, _ = flattened_tests[flattened_position]
            row = rows[row_index]
            row["ground_truth_tests"][test_index]["test_quartile"] = quartile
            row["ground_truth_tests"][test_index]["test_group_name"] = f"{dataset_stub}_test_quartile{quartile}"
            row["test_summaries"][test_index]["test_quartile"] = quartile
            row["test_summaries"][test_index]["test_group_name"] = f"{dataset_stub}_test_quartile{quartile}"
            row["ground_truth"] = json.dumps(row["ground_truth_tests"])

    return rows


def save_and_optionally_push(out_ds: Dataset, args: argparse.Namespace, split: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    local_output_dir = os.path.join(args.save_local_dir, split, timestamp)
    os.makedirs(local_output_dir, exist_ok=True)
    out_ds.save_to_disk(local_output_dir)
    local_jsonl_path = os.path.join(local_output_dir, f"{split}.jsonl")
    out_ds.to_json(local_jsonl_path)
    logger.info("Saved local dataset artifacts for %s to %s (jsonl: %s)", split, local_output_dir, local_jsonl_path)

    if args.push_to_hub:
        logger.info("Pushing split %s to %s", split, args.push_to_hub)
        out_ds.push_to_hub(args.push_to_hub, split=split, private=args.private)


def main() -> None:
    args = parse_args()
    logger.info("Loading dataset %s[%s]", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, num_proc=max_num_processes())
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts, generation_messages = build_prompts(ds, args.model, args.chat_template)
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )
    completions_by_prompt = generate_completions(
        prompts=prompts,
        model=args.model,
        num_engines=args.num_engines,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        sampling_params=sampling_params,
    )
    rows = build_output_rows(ds, generation_messages, completions_by_prompt, args, args.split)
    out_ds = Dataset.from_list(rows)
    save_and_optionally_push(out_ds, args, args.split)


if __name__ == "__main__":
    main()
