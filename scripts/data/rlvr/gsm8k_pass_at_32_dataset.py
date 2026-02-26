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

from datasets import Dataset, load_dataset
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


def build_prompts(ds: Dataset, model_name: str, chat_template_name: str) -> list[str]:
    if chat_template_name not in CHAT_TEMPLATES:
        raise ValueError(f"Unknown chat template: {chat_template_name}")

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.chat_template = CHAT_TEMPLATES[chat_template_name]

    prompts = [
        tok.apply_chat_template(sample["messages"], add_generation_prompt=True, tokenize=False) for sample in ds
    ]
    return prompts


def main() -> None:
    args = parse_args()
    logger.info("Loading dataset %s[%s]", args.dataset, args.split)
    ds = load_dataset(args.dataset, split=args.split, num_proc=max_num_processes())
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    prompts = build_prompts(ds, args.model, args.chat_template)

    logger.info(
        "Initializing vLLM model=%s tp=%d rows=%d samples=%d",
        args.model,
        args.tensor_parallel_size,
        len(ds),
        args.num_samples,
    )
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
    )
    sampling_params = SamplingParams(
        n=args.num_samples, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed
    )

    logger.info("Generating completions")
    outputs = llm.generate(prompts, sampling_params)

    verifier = GSM8KVerifier()
    rows: list[dict] = []

    logger.info("Scoring completions and building output rows")
    for sample, request_output in zip(ds, outputs):
        completions = [completion.text for completion in request_output.outputs]
        pass_count = 0
        for completion in completions:
            score = verifier(tokenized_prediction=[], prediction=completion, label=str(sample["ground_truth"])).score
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
