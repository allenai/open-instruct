"""Generate pass@k completions for AIME-style datasets and optionally push them to HF Hub.

Example:
uv run python scripts/data/rlvr/aime_pass_at_k_dataset.py \
  --dataset allenai/aime2024-25-rlvr \
  --splits test_2024 test_2025 \
  --model allenai/Olmo-3-1025-7B \
  --chat-template olmo_thinker_rlzero \
  --num-samples 32 \
  --tensor-parallel-size 1 \
  --num-engines 8 \
  --push-to-hub <entity>/aime2024-25-rlvr-olmo3-7b-base-pass32
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Any

import ray
from datasets import Dataset, load_dataset
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES
from open_instruct.ground_truth_utils import MathVerifier
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pass@k completions for AIME splits and build a Hub dataset.")
    parser.add_argument("--dataset", default="allenai/aime2024-25-rlvr", help="Input HF dataset")
    parser.add_argument("--splits", nargs="+", default=["test_2024", "test_2025"], help="Input splits")
    parser.add_argument("--model", default="allenai/Olmo-3-1025-7B", help="Model used for generation")
    parser.add_argument("--chat-template", default="olmo_thinker_rlzero", help="Chat template name")
    parser.add_argument("--num-samples", type=int, default=32, help="Completions per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max new tokens per completion")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument(
        "--num-engines",
        "--num_engines",
        dest="num_engines",
        type=int,
        default=8,
        help="Number of independent vLLM engines to run in parallel",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process per split")
    parser.add_argument(
        "--save-local-dir",
        default="/tmp/aime_pass_at_k_dataset_outputs",
        help="Base directory for saving generated dataset artifacts before push",
    )
    parser.add_argument(
        "--push-to-hub",
        default="",
        help="Output HF dataset repo; set empty string to skip push",
    )
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


def _generate_completions_single_engine(
    prompts: list[str],
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    sampling_params: SamplingParams,
) -> list[list[str]]:
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [[completion.text for completion in request_output.outputs] for request_output in outputs]


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
    sampling_params_kwargs: dict,
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


def _generate_completions_multi_engine(
    prompts: list[str],
    model: str,
    num_engines: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    sampling_params: SamplingParams,
) -> list[list[str]]:
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


def build_output_rows(
    ds: Dataset,
    generation_messages: list[list[dict[str, str]]],
    completions_by_prompt: list[list[str]],
    args: argparse.Namespace,
    split: str,
) -> list[dict[str, Any]]:
    verifier = MathVerifier()
    rows: list[dict[str, Any]] = []

    logger.info("Scoring completions and building output rows for %s", split)
    for sample, sanitized_messages, completions in zip(ds, generation_messages, completions_by_prompt):
        pass_count = 0
        for completion in completions:
            score = verifier(tokenized_prediction=[], prediction=completion, label=str(sample["ground_truth"])).score
            pass_count += int(score)

        row = dict(sample)
        row["messages"] = sanitized_messages
        row["completions"] = completions
        row["pass_count"] = pass_count
        row["pass_rate"] = f"{pass_count}/{args.num_samples}"
        row["num_samples"] = args.num_samples
        row["generator_model"] = args.model
        row["generator_chat_template"] = args.chat_template
        row["generator_temperature"] = args.temperature
        row["generator_top_p"] = args.top_p
        row["generator_max_tokens"] = args.max_tokens
        row["source_split"] = split
        rows.append(row)

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
        logger.info("Pushing %s[%s] to hub", args.push_to_hub, split)
        try:
            out_ds.push_to_hub(args.push_to_hub, split=split, private=args.private)
            logger.info("Finished pushing split %s", split)
        except Exception:
            logger.exception(
                "Push failed for split %s. Local artifacts are available at %s and can be reused for a later push.",
                split,
                local_output_dir,
            )
            raise


def main() -> None:
    args = parse_args()
    if args.num_engines < 1:
        raise ValueError(f"--num-engines must be >= 1, got {args.num_engines}")

    sampling_params = SamplingParams(
        n=args.num_samples, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
    )

    for split in args.splits:
        logger.info("Loading dataset %s[%s]", args.dataset, split)
        ds = load_dataset(args.dataset, split=split, num_proc=max_num_processes())
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))

        prompts, generation_messages = build_prompts(ds, args.model, args.chat_template)
        logger.info(
            "Initializing vLLM model=%s engines=%d tp=%d split=%s rows=%d samples=%d",
            args.model,
            args.num_engines,
            args.tensor_parallel_size,
            split,
            len(ds),
            args.num_samples,
        )

        logger.info("Generating completions for %s", split)
        if args.num_engines == 1:
            completions_by_prompt = _generate_completions_single_engine(
                prompts=prompts,
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                sampling_params=sampling_params,
            )
        else:
            completions_by_prompt = _generate_completions_multi_engine(
                prompts=prompts,
                model=args.model,
                num_engines=args.num_engines,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                sampling_params=sampling_params,
            )

        rows = build_output_rows(ds, generation_messages, completions_by_prompt, args, split)
        out_ds = Dataset.from_list(rows)
        logger.info("Built output dataset for %s with %d rows and columns: %s", split, len(out_ds), out_ds.column_names)
        save_and_optionally_push(out_ds, args, split)

    if not args.push_to_hub:
        logger.info("Skipping push_to_hub because --push-to-hub was empty")


if __name__ == "__main__":
    main()
