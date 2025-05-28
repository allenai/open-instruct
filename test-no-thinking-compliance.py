#!/usr/bin/env python
"""
Compare two system prompts on two HF datasets with vLLM.

• Randomly sample up-to 10 000 rows from each dataset.
• Create *two* versions of every example (one per system prompt).
• Run them through a vLLM model in batches.
• Post-process each generation with a user-supplied parse function.
• Print aggregate metrics.

Dependencies:
    pip install "vllm>=0.4.0" datasets tqdm transformers accelerate
"""

from __future__ import annotations
import argparse
import random
import time
from typing import List, Dict, Any

from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


# --------------------------------------------------------------------------- #
#                             CONFIGURATION                                   #
# --------------------------------------------------------------------------- #

SYSTEM_PROMPTS: List[str] = [
    "thinking out loud enabled",
    "thinking out loud disabled",
]

MODEL_NAME   = "allenai/open_instruct_dev"  # repository / path
MODEL_REV    = "qwen-2-tulu-general-thinker-rewrite-on-off-test-160k__123__1747894780"

NUM_SAMPLES  = 1000
DATASETS = {
    "tulu3_rewritten":
        ("hamishivi/tulu_3_rewritten_100k",  "messages"),  # column name
    "tulu3_wildchat_unused":
        ("allenai/tulu-3-wildchat-unused",   "prompt"),
}

BATCH_SIZE = 128

SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
)

# --------------------------------------------------------------------------- #
#                               UTILITIES                                     #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)


def load_and_sample(dataset_name: str, column: str,
                    k: int) -> list[dict[str, Any]]:
    ds = load_dataset(dataset_name,
                      split="train",
                      trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(k, len(ds))))
    print(f"Loaded {dataset_name}: {len(ds)} examples")
    return ds


# ---------- prompt building ------------------------------------------------- #
def build_prompt(tokenizer: AutoTokenizer,
                 system_prompt: str,
                 body: Any,
                 body_type: str) -> str:
    """Return a single prompt string ready for vLLM.generate()."""
    messages = [{"role": "system", "content": system_prompt}]

    if body_type == "messages":          # already a list[dict]
        messages.extend(body)
    elif body_type == "prompt":
        messages.append({"role": "user", "content": body})
    else:
        raise ValueError(f"Unknown body_type: {body_type}")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,      # inserts assistant tag
    )


# ---------- analyse / parse ------------------------------------------------- #
def parse_output(text: str) -> Dict[str, float]:
    """Toy evaluator: detect presence of `</think>` tag."""
    closed = "</think>" in text
    contains = closed and (len(text.split("</think>")[0]) > 0)
    return {
        "closed_thinking_tag": float(closed),
        "contains_thoughts":   float(contains),
    }


# --------------------------------------------------------------------------- #
#                                   MAIN                                      #
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    t0 = time.time()
    set_seed()

    # 1. Load datasets --------------------------------------------------------
    sampled_sets = {
        tag: load_and_sample(dname, col, NUM_SAMPLES)
        for tag, (dname, col) in DATASETS.items()
    }

    # 2. Tokenizer (needed for chat_template) ---------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REV)

    # 3. Build prompt list ----------------------------------------------------
    prompts: List[str]      = []
    meta_info: List[tuple]  = []   # (dataset_tag, system_idx)

    for sys_idx, sys_prompt in enumerate(SYSTEM_PROMPTS):
        for ds_tag, ds in sampled_sets.items():
            col_name = DATASETS[ds_tag][1]
            for row in ds:
                prompt_str = build_prompt(
                    tokenizer,
                    sys_prompt,
                    row[col_name],
                    col_name
                )
                prompts.append(prompt_str)
                meta_info.append((ds_tag, sys_idx))

    print(f"Total generations to run: {len(prompts):,}")

    # 4. Instantiate vLLM -----------------------------------------------------
    llm = LLM(
        model=MODEL_NAME,
        revision=MODEL_REV,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    # 5. Batched inference ----------------------------------------------------
    generations: List[str] = [None] * len(prompts)

    for start in tqdm(range(0, len(prompts), BATCH_SIZE),
                    desc="vLLM inference"):
        batch_prompts = prompts[start:start + BATCH_SIZE]
        outs = llm.generate(batch_prompts, SAMPLING_PARAMS)

        # outs[i] corresponds to batch_prompts[i]
        for i, out in enumerate(outs):
            generations[start + i] = out.outputs[0].text

    # 6. Parse & aggregate ----------------------------------------------------
    aggregates: Dict[str, Dict[str, float]] = {}
    counts:     Dict[str, int]             = {}

    for (ds_tag, sys_idx), text in zip(meta_info, generations):
        metrics = parse_output(text)
        key = f"{ds_tag}|sys{sys_idx}"

        if key not in aggregates:
            aggregates[key] = {k: 0.0 for k in metrics}
            counts[key] = 0

        for k, v in metrics.items():
            aggregates[key][k] += v
        counts[key] += 1

    # 7. Report ---------------------------------------------------------------
    print("\n=== Aggregate results ===")
    for ds_tag in sampled_sets.keys():
        for sys_idx, sys_prompt in enumerate(SYSTEM_PROMPTS):
            key = f"{ds_tag}|sys{sys_idx}"
            n = counts[key]
            print(f"\nDataset: {ds_tag:25} | System prompt #{sys_idx}:")
            for metric, total in aggregates[key].items():
                print(f"  {metric:20}: {total / n:.4f} (avg over {n})")

    print(f"\nFinished in {(time.time() - t0) / 60:.1f} minutes")


# --------------------------------------------------------------------------- #
#                                   CLI                                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallelism for vLLM.")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16"],
                        help="Model dtype.")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="Fraction of GPU memory vLLM may use.")
    args = parser.parse_args()
    main(args)