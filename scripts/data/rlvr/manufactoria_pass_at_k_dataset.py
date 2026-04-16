"""Generate or recompute Manufactoria pass@k metrics and difficulty for RLVR datasets.

Adds a `difficulty` column: length-N list (one entry per ground-truth test), integers 1–4.
Difficulty 1 is reserved for tests with zero per-test solve rate across the k completions.
The remaining non-zero tests are split into three dataset-level buckets (2 = harder, 4 = easier).
Also stores explicit full-pass and per-test pass metrics so difficulty can be recomputed from an
existing dataset that already contains a `completions` column.

Requires a running Manufactoria API (same as training). Set MANUFACTORIA_API_URL or pass --manufactoria-api-url.

Example (local generation):
  export MANUFACTORIA_API_URL=http://localhost:1235
  uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \\
    --dataset manufactoria/has_test \\
    --split train \\
    --model Qwen/Qwen3-4B-Instruct-2507 \\
    --chat-template from_model \\
    --num-samples 32 \\
    --max_prompt_token_length 2048 \\
    --response_length 8192 \\
    --tensor-parallel-size 1 \\
    --num-engines 8

Recalculate pass metrics and difficulty from an existing HF dataset that already has a `completions`
column (no vLLM), then either dry-run and print scores or push:
  uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \\
    --dataset mnoukhov/manufactoria-qwen3-4b-instruct-warmup650-pass128 \\
    --split train --use-existing-completions --dry-run

  uv run python scripts/data/rlvr/manufactoria_pass_at_k_dataset.py \\
    --dataset mnoukhov/manufactoria-qwen3-4b-instruct-warmup650-pass128 \\
    --split train --use-existing-completions \\
    --push-to-hub mnoukhov/manufactoria-qwen3-4b-instruct-warmup650-pass128-rescored
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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
from open_instruct.dataset_transformation import CHAT_TEMPLATES
from open_instruct.ground_truth_utils import ManufactoriaVerifier, ManufactoriaVerifierConfig
from open_instruct.utils import max_num_processes

logger = logger_utils.setup_logger(__name__)

_DEFAULT_MANUFACTORIA_URL = os.environ.get("MANUFACTORIA_API_URL", "http://localhost:1235") + "/test_solution"
DIFFICULTY_KEY = "difficulty"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate or recompute Manufactoria pass@k metrics and difficulty.")
    parser.add_argument(
        "--dataset", default="manufactoria/has_train", help="Input HF dataset (e.g. manufactoria/has_test)"
    )
    parser.add_argument("--split", default="train", help="Input split name")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507", help="Model used for generation")
    parser.add_argument(
        "--chat-template",
        default="from_model",
        help="Chat template: 'from_model' uses the tokenizer checkpoint template, or a key from CHAT_TEMPLATES",
    )
    parser.add_argument("--num-samples", type=int, default=32, help="Completions per prompt (k)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--response_length", type=int, default=8192, help="Max new tokens per completion")
    parser.add_argument(
        "--max_prompt_token_length", type=int, default=2048, help="Prompt token budget used to set vLLM max_model_len"
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="vLLM tensor parallel size")
    parser.add_argument(
        "--num-engines",
        "--num_engines",
        dest="num_engines",
        type=int,
        default=1,
        help="Number of independent vLLM engines (Ray workers) for data-parallel generation",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="vLLM GPU memory utilization")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--limit", type=int, default=None, help="Optional max rows to process")
    parser.add_argument(
        "--use-existing-completions",
        action="store_true",
        help=(
            "Load an existing HF dataset that already has a `completions` column; call the Manufactoria "
            "API to recompute Full pass / per-test pass metrics and (unless --no-difficulty) difficulty. "
            "Skips vLLM generation."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "After scoring, print aggregate and per-row sample scores only; do not save to disk or push to hub. "
            "Typical with --use-existing-completions to verify recomputed metrics before uploading."
        ),
    )
    parser.add_argument(
        "--manufactoria-api-url",
        default=_DEFAULT_MANUFACTORIA_URL,
        help="Manufactoria test_solution endpoint (default: $MANUFACTORIA_API_URL/test_solution)",
    )
    parser.add_argument("--manufactoria-max-execution-time", type=float, default=1.0, help="Per-test API time limit")
    parser.add_argument(
        "--manufactoria-scoring-mode",
        default="pass_rate",
        choices=["all_pass", "pass_rate"],
        help="Verifier scoring mode (match training; qwen3_4b_phase1_has_8gpu uses pass_rate)",
    )
    parser.add_argument(
        "--pass-score-threshold",
        type=float,
        default=1.0,
        help="Count a completion as correct if verifier score >= this (1.0 = all tests passed for pass_rate mode)",
    )
    parser.add_argument(
        "--score-threads",
        type=int,
        default=32,
        help="Max parallel HTTP verifier calls per prompt when scoring completions",
    )
    parser.add_argument(
        "--save-local-dir",
        default="/tmp/manufactoria_pass_at_k_outputs",
        help="Base directory for saving generated dataset artifacts before push",
    )
    parser.add_argument(
        "--push-to-hub",
        default="",
        help=(
            "HF dataset repo id to push (e.g. your-org/manufactoria-has-train-qwen3-4b-instruct-pass32). "
            "Empty string skips push. The qwen3_4b shell wrapper sets defaults for train/test."
        ),
    )
    parser.add_argument("--private", action="store_true", help="Push dataset as private (default public)")
    parser.add_argument(
        "--no-difficulty",
        action="store_true",
        help=f"Do not add {DIFFICULTY_KEY!r} (per-test quartiles 1–4 from pass@k rollouts)",
    )
    return parser.parse_args()


def normalize_manufactoria_ground_truth(raw: Any) -> Any:
    """Match RLVR layout: list of tests, or wrapped [[tests...]] after tokenization."""
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return raw
        return normalize_manufactoria_ground_truth(parsed)
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], list):
            return raw[0]
        return raw
    return raw


def build_prompts(ds: Dataset, model_name: str, chat_template_name: str) -> list[str]:
    tok = AutoTokenizer.from_pretrained(model_name)
    if chat_template_name != "from_model":
        if chat_template_name not in CHAT_TEMPLATES:
            raise ValueError(f"Unknown chat template: {chat_template_name}. Use from_model or a CHAT_TEMPLATES key.")
        tok.chat_template = CHAT_TEMPLATES[chat_template_name]
    elif not getattr(tok, "chat_template", None):
        raise ValueError("Tokenizer has no chat_template; set --chat-template to a CHAT_TEMPLATES key.")

    prompts = []
    for sample in ds:
        if "messages" not in sample:
            raise KeyError('Expected column "messages" in dataset rows')
        prompts.append(tok.apply_chat_template(sample["messages"], add_generation_prompt=True, tokenize=False))
    return prompts


def _generate_completions_single_engine(
    prompts: list[str],
    model: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    sampling_params: SamplingParams,
    max_model_len: int,
) -> list[list[str]]:
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
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
    max_model_len: int,
) -> tuple[int, list[list[str]]]:
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend="uni" if tensor_parallel_size == 1 else "mp",
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        enable_prefix_caching=True,
        max_model_len=max_model_len,
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
    max_model_len: int,
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
                placement_group=pg, placement_group_capture_child_tasks=True, placement_group_bundle_index=bundle_index
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
                    max_model_len=max_model_len,
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


def _num_tests_in_label(label: Any) -> int:
    if isinstance(label, list):
        return len(label)
    return 0


def _align_per_test_pass(vec: list[float] | None, n_tests: int) -> list[float]:
    if n_tests <= 0:
        return []
    if not vec:
        return [0.0] * n_tests
    out = [float(x) for x in vec]
    if len(out) < n_tests:
        out.extend([0.0] * (n_tests - len(out)))
    elif len(out) > n_tests:
        out = out[:n_tests]
    return out


def _format_pass_rate(pass_count: int, num_samples: int) -> str:
    return f"{pass_count}/{num_samples}"


def _extract_per_test_pass_vector(result_metadata: dict[str, Any], n_tests: int) -> list[float]:
    test_results = result_metadata.get("manufactoria_test_results", []) or []
    if n_tests <= 0:
        return []

    per_test_pass = [0.0] * n_tests
    for test_result in test_results:
        if not isinstance(test_result, dict):
            continue
        test_index = test_result.get("test_index")
        if isinstance(test_index, int) and 0 <= test_index < n_tests:
            per_test_pass[test_index] = float(bool(test_result.get("passed")))
    return per_test_pass


def _per_test_pass_rates_from_row(row: dict[str, Any]) -> list[float]:
    """Per-test pass rate in [0, 1] for each test; empty if missing counts or num_samples."""
    ptc = row.get("Per-test pass count")
    if not isinstance(ptc, list) or not ptc:
        return []
    num_samples = int(row.get("num_samples", 0) or 0)
    if num_samples <= 0:
        return []
    return [float(c) / float(num_samples) for c in ptc]


def print_rescored_dataset_summary(rows: list[dict[str, Any]], sample_row_limit: int = 8) -> None:
    """Log aggregate stats and a short sample of rows (for --dry-run)."""
    n = len(rows)
    if n == 0:
        logger.info("dry-run summary: empty dataset")
        return

    total_completions = 0
    total_full_pass_completions = 0
    difficulty_flat: list[int] = []
    weighted_rate_sum = 0.0
    weighted_rate_count = 0

    for row in rows:
        ns = int(row.get("num_samples", 0) or 0)
        total_completions += ns
        total_full_pass_completions += int(row.get("Full pass count", 0) or 0)
        diff = row.get(DIFFICULTY_KEY)
        if isinstance(diff, list):
            for x in diff:
                try:
                    difficulty_flat.append(int(x))
                except (TypeError, ValueError):
                    continue
        for r in _per_test_pass_rates_from_row(row):
            weighted_rate_sum += r
            weighted_rate_count += 1

    logger.info("dry-run summary: rows=%d", n)
    if total_completions > 0:
        logger.info(
            "dry-run summary: aggregate_full_pass_rate=%.6f (%d full passes / %d completions)",
            total_full_pass_completions / total_completions,
            total_full_pass_completions,
            total_completions,
        )
    if weighted_rate_count > 0:
        logger.info(
            "dry-run summary: mean_per_test_pass_rate_over_all_tests=%.6f (%d test slots)",
            weighted_rate_sum / weighted_rate_count,
            weighted_rate_count,
        )
    if difficulty_flat:
        ctr = Counter(difficulty_flat)
        logger.info(
            "dry-run summary: difficulty_bucket_counts=%s (1=unsolved test … 4=easier bucket)",
            dict(sorted(ctr.items())),
        )

    shown = min(sample_row_limit, n)
    logger.info(
        "dry-run sample: first %d rows (per_test_pass_rates is one float per ground-truth test, order matches tests)",
        shown,
    )
    for i in range(shown):
        row = rows[i]
        rates = _per_test_pass_rates_from_row(row)
        ptc = row.get("Per-test pass count")
        n_pt = len(ptc) if isinstance(ptc, list) else 0
        logger.info(
            "  row %d: num_samples=%s full_pass_count=%s per_test_pass_count_len=%d per_test_pass_rates=%s",
            i,
            row.get("num_samples"),
            row.get("Full pass count"),
            n_pt,
            rates,
        )


def assign_global_difficulty_quartiles(rows: list[dict[str, Any]]) -> None:
    """Assign dataset-level difficulty buckets from per-test pass rates across all rows.

    Difficulty 1 is assigned to every zero-pass test and only zero-pass tests.
    Remaining tests are split into three stable buckets with difficulties 2, 3, and 4.
    """
    indexed_rates: list[tuple[int, int, float]] = []
    for row_idx, row in enumerate(rows):
        pass_counts = row.get("Per-test pass count", []) or []
        num_samples = int(row.get("num_samples", 0) or 0)
        if num_samples <= 0:
            row[DIFFICULTY_KEY] = [1] * len(pass_counts)
            continue
        for test_idx, pass_count in enumerate(pass_counts):
            indexed_rates.append((row_idx, test_idx, float(pass_count) / float(num_samples)))

    if not indexed_rates:
        return

    for row in rows:
        row[DIFFICULTY_KEY] = [1] * len(row.get("Per-test pass count", []) or [])

    nonzero_rates = [(row_idx, test_idx, rate) for row_idx, test_idx, rate in indexed_rates if rate > 0.0]
    if not nonzero_rates:
        return

    ordered_nonzero_rates = sorted(nonzero_rates, key=lambda item: item[2])
    bucket_size = math.ceil(len(ordered_nonzero_rates) / 3)
    for bucket_idx, start in enumerate(range(0, len(ordered_nonzero_rates), bucket_size), start=2):
        difficulty = min(bucket_idx, 4)
        for row_idx, test_idx, _rate in ordered_nonzero_rates[start : start + bucket_size]:
            rows[row_idx][DIFFICULTY_KEY][test_idx] = difficulty


def _score_completions(
    verifier: ManufactoriaVerifier,
    label: Any,
    completions: list[str],
    threshold: float,
    max_workers: int,
    with_difficulty: bool,
) -> tuple[int, list[int]]:
    n_tests = _num_tests_in_label(label)

    def score_one(text: str) -> tuple[int, list[float] | None]:
        result = verifier([], text, label)
        passed = int(float(result.score) >= threshold - 1e-8)
        if not with_difficulty or n_tests <= 0:
            return passed, None
        meta = getattr(result, "metadata", {}) or {}
        vec = _extract_per_test_pass_vector(meta, n_tests)
        return passed, vec

    workers = min(max_workers, max(1, len(completions)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(score_one, completions))

    pass_count = sum(r[0] for r in results)
    if not with_difficulty or n_tests <= 0:
        return pass_count, []

    matrix = np.array([r[1] for r in results], dtype=float)
    per_test_pass_count = matrix.sum(axis=0).astype(int).tolist()
    return pass_count, per_test_pass_count


def _row_num_samples(completions: list[str], fallback_num_samples: int) -> int:
    return len(completions) if completions else fallback_num_samples


def build_output_row(
    sample: dict[str, Any], completions: list[str], verifier: ManufactoriaVerifier, args: argparse.Namespace
) -> dict[str, Any]:
    if "ground_truth" not in sample:
        raise KeyError('Expected column "ground_truth" in dataset rows')

    label = normalize_manufactoria_ground_truth(sample["ground_truth"])
    with_difficulty = not args.no_difficulty
    full_pass_count, per_test_pass_count = _score_completions(
        verifier, label, completions, args.pass_score_threshold, args.score_threads, with_difficulty=with_difficulty
    )

    num_samples = _row_num_samples(completions, args.num_samples)
    row = {
        **sample,
        "completions": completions,
        "Full pass count": full_pass_count,
        "Full pass rate": _format_pass_rate(full_pass_count, num_samples),
        "num_samples": num_samples,
        "generator_manufactoria_scoring_mode": args.manufactoria_scoring_mode,
        "generator_pass_score_threshold": args.pass_score_threshold,
    }
    if not args.use_existing_completions:
        row["generator_model"] = args.model
        row["generator_chat_template"] = args.chat_template
        row["generator_temperature"] = args.temperature
        row["generator_top_p"] = args.top_p
        row["generator_max_tokens"] = args.response_length
    if with_difficulty:
        row["Per-test pass count"] = per_test_pass_count
        row["Per-test pass rate"] = [_format_pass_rate(count, num_samples) for count in per_test_pass_count]
    return row


def main() -> None:
    args = parse_args()
    if args.num_engines < 1:
        raise ValueError(f"--num-engines must be >= 1, got {args.num_engines}")

    logger.info("Loading dataset %s[%s]", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, num_proc=max_num_processes())
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    if args.use_existing_completions:
        if "completions" not in ds.column_names:
            raise KeyError('Expected column "completions" when using --use-existing-completions')
        completions_by_prompt = [list(sample["completions"]) for sample in ds]
        logger.info("Reusing existing completions from dataset rows=%d", len(ds))
    else:
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
            n=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.response_length,
            seed=args.seed,
        )
        max_model_len = args.max_prompt_token_length + args.response_length
        logger.info(
            "Using max_model_len=%d (max_prompt_token_length=%d + response_length=%d)",
            max_model_len,
            args.max_prompt_token_length,
            args.response_length,
        )

        logger.info("Generating completions")
        if args.num_engines == 1:
            completions_by_prompt = _generate_completions_single_engine(
                prompts=prompts,
                model=args.model,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                sampling_params=sampling_params,
                max_model_len=max_model_len,
            )
        else:
            completions_by_prompt = _generate_completions_multi_engine(
                prompts=prompts,
                model=args.model,
                num_engines=args.num_engines,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                sampling_params=sampling_params,
                max_model_len=max_model_len,
            )

    verifier = ManufactoriaVerifier(
        ManufactoriaVerifierConfig(
            manufactoria_api_url=args.manufactoria_api_url,
            manufactoria_max_execution_time=args.manufactoria_max_execution_time,
            manufactoria_scoring_mode=args.manufactoria_scoring_mode,
        )
    )

    rows: list[dict] = []
    logger.info("Scoring completions via Manufactoria API (%s)", args.manufactoria_api_url)
    for sample, completions in zip(ds, completions_by_prompt):
        rows.append(build_output_row(sample, completions, verifier, args))

    if not args.no_difficulty:
        assign_global_difficulty_quartiles(rows)

    if args.dry_run:
        print_rescored_dataset_summary(rows)
        logger.info("Dry run: skipping save_to_disk and push_to_hub")
        return

    out_ds = Dataset.from_list(rows)
    logger.info("Built output dataset with %d rows and columns: %s", len(out_ds), out_ds.column_names)

    safe_ds_name = args.dataset.replace("/", "__")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    local_output_dir = os.path.join(args.save_local_dir, f"{safe_ds_name}_{args.split}_{timestamp}")
    os.makedirs(local_output_dir, exist_ok=True)
    out_ds.save_to_disk(local_output_dir)
    local_jsonl_path = os.path.join(local_output_dir, f"{args.split}.jsonl")
    out_ds.to_json(local_jsonl_path)
    logger.info("Saved local dataset artifacts to %s (jsonl: %s)", local_output_dir, local_jsonl_path)

    if args.push_to_hub:
        logger.info("Pushing dataset to hub: %s (split=%s)", args.push_to_hub, args.split)
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
        logger.info("Skipping push_to_hub (empty --push-to-hub)")


if __name__ == "__main__":
    main()
