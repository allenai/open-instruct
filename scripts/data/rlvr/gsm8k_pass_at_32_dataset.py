"""Generate pass@k completions for GSM8K-style datasets and optionally push to HF Hub.

Example:
uv run python scripts/data/rlvr/gsm8k_pass_at_32_dataset.py \
  --dataset mnoukhov/gsm8k-platinum-openinstruct \
  --split test \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --chat-template qwen_instruct_gsm8k \
  --num-samples 32 \
  --tensor-parallel-size 2 \
  --push-to-hub mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct
"""

import argparse
import os
from datetime import datetime, timezone

import ray
from datasets import Dataset, load_dataset
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct import logger_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES
from open_instruct.ground_truth_utils import GSM8KVerifier
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pass@k completions and build a Hub dataset.")
    parser.add_argument("--dataset", default="mnoukhov/gsm8k-platinum-openinstruct", help="Input HF dataset")
    parser.add_argument("--split", default="test", help="Input split")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model used for generation")
    parser.add_argument("--chat-template", default="qwen_instruct_boxed_math", help="Chat template name")
    parser.add_argument("--num-samples", type=int, default=32, help="Completions per prompt")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max new tokens per completion")
    parser.add_argument("--tensor-parallel-size", type=int, default=2, help="vLLM tensor parallel size")
    parser.add_argument(
        "--num-engines",
        "--num_engines",
        dest="num_engines",
        type=int,
        default=1,
        help="Number of independent vLLM engines to run in parallel",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to process")
    parser.add_argument(
        "--save-local-dir",
        default="/tmp/gsm8k_pass_at_32_dataset_outputs",
        help="Base directory for saving generated dataset artifacts before push",
    )
    parser.add_argument(
        "--push-to-hub",
        default="mnoukhov/gsm8k-platinum-openinstruct-0.5b-instruct",
        help="Output HF dataset repo; set empty string to skip push",
    )
    parser.add_argument("--private", action="store_true", help="Push dataset as private (default public)")
    return parser.parse_args()


def normalize_gsm8k_ground_truth(ground_truth: str | list[str]) -> str:
    if isinstance(ground_truth, list):
        if not ground_truth:
            raise ValueError("ground_truth list is empty")
        return str(ground_truth[0])
    return str(ground_truth)


def build_prompts(ds: Dataset, model_name: str, chat_template_name: str) -> list[str]:
    if chat_template_name not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template: {chat_template_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.chat_template = CHAT_TEMPLATES[chat_template_name]

    prompts = [
        tok.apply_chat_template(sample["messages"], add_generation_prompt=True, tokenize=False) for sample in ds
    ]
    return prompts


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


def main() -> None:
    args = parse_args()
    if args.num_engines < 1:
        raise ValueError(f"--num-engines must be >= 1, got {args.num_engines}")

    logger.info("Loading dataset %s[%s]", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, num_proc=max_num_processes())
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = build_prompts(ds, args.model, args.chat_template)

    logger.info(
        "Initializing vLLM model=%s engines=%d tp=%d rows=%d samples=%d",
        args.model,
        args.num_engines,
        args.tensor_parallel_size,
        len(ds),
        args.num_samples,
    )
    sampling_params = SamplingParams(
        n=args.num_samples, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
    )

    logger.info("Generating completions")
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

    verifier = GSM8KVerifier()
    rows: list[dict] = []

    logger.info("Scoring completions and building output rows")
    for sample, completions in zip(ds, completions_by_prompt):
        label = normalize_gsm8k_ground_truth(sample["ground_truth"])
        pass_count = 0
        for completion in completions:
            score = verifier(tokenized_prediction=[], prediction=completion, label=label).score
            pass_count += int(score)

        rows.append(
            {
                **sample,
                "completions": completions,
                "pass_count": pass_count,
                "pass_rate": f"{pass_count}/{args.num_samples}",
                "num_samples": args.num_samples,
                "generator_model": args.model,
                "generator_chat_template": args.chat_template,
                "generator_temperature": args.temperature,
                "generator_top_p": args.top_p,
                "generator_max_tokens": args.max_tokens,
            }
        )

    out_ds = Dataset.from_list(rows)
    logger.info("Built output dataset with %d rows and columns: %s", len(out_ds), out_ds.column_names)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    local_output_dir = os.path.join(args.save_local_dir, f"{args.split}_{timestamp}")
    os.makedirs(local_output_dir, exist_ok=True)
    out_ds.save_to_disk(local_output_dir)
    local_jsonl_path = os.path.join(local_output_dir, f"{args.split}.jsonl")
    out_ds.to_json(local_jsonl_path)
    logger.info("Saved local dataset artifacts to %s (jsonl: %s)", local_output_dir, local_jsonl_path)

    if args.push_to_hub:
        logger.info("Pushing dataset to hub: %s", args.push_to_hub)
        try:
            out_ds.push_to_hub(args.push_to_hub, split=args.split, private=args.private)
            logger.info("Finished pushing dataset")
        except Exception:
            logger.exception(
                "Push failed. Local artifacts are available at %s and can be reused for a later push.",
                local_output_dir,
            )
            raise
    else:
        logger.info("Skipping push_to_hub because --push-to-hub was empty")


if __name__ == "__main__":
    main()
