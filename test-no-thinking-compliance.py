#!/usr/bin/env python
"""
Run 10k random examples from two HF datasets through a chat model with two
different system prompts, then analyse the generations.

Requires:
    pip install --upgrade vllm datasets tqdm accelerate

Author: <you>
"""

from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
import json
import time

import datasets
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import random_uuid
from vllm.entrypoints.openai.protocol import ChatCompletionRequest


# --------------------------------------------------------------------------- #
# --------------------------- CONFIGURATION ---------------------------------- #
# --------------------------------------------------------------------------- #

# 1. The two system prompts you want to compare
SYSTEM_PROMPTS: List[str] = [
    "thinking out loud enabled",
    "thinking out loud disabled"
]

# 2. Which model to load? (Anything with a chat template – 8-bit loading works)
MODEL_NAME: str = "allenai/open_instruct_dev"     # change if needed
MODEL_REVISION: str = "qwen-2-tulu-general-thinker-rewrite-on-off-test-160k__123__1747894780"

# 3. Inference parameters
SAMPLING_PARAMS = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=512,
    stop=None,
)

# 4. Data
NUM_SAMPLES: int = 10_000
DATASETS = {
    "tulu3_rewritten":         ("hamishivi/tulu_3_rewritten_100k", "messages"),
    "tulu3_wildchat_unused":   ("allenai/tulu-3-wildchat-unused",                                "prompt"),
}

# 5. Batch size for vLLM.generate(); adapt to your GPU memory
BATCH_SIZE = 32

# --------------------------------------------------------------------------- #
# ----------------------------- UTILITIES ----------------------------------- #
# --------------------------------------------------------------------------- #

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    datasets.utils.logging.set_verbosity_error()


def load_and_sample(dataset_name: str, column: str, num_rows: int) -> datasets.Dataset:
    """Load one split (default 'train') and return *num_rows* random rows."""
    ds = load_dataset(dataset_name, split='train', trust_remote_code=True)
    ds = ds.shuffle(seed=42).select(range(min(num_rows, len(ds))))
    print(f"Loaded {dataset_name}: {len(ds)} examples")
    return ds


def hf_messages_to_chat(messages: List[Dict[str, Any]]) -> List[ChatMessage]:
    """Convert HF message dicts (role/content) → vLLM ChatMessage list."""
    chat: List[ChatMessage] = []
    assert messages[0]["role"] == "user"
    chat.append(ChatMessage(role="user", content=messages[0]["content"]))
    return chat


def make_msg(role: str, content: str) -> dict:
    """Return an OpenAI-compatible chat message dict."""
    return {"role": role, "content": content}

def build_chat_request(system_prompt: str,
                       body: Any,
                       body_type: str) -> ChatCompletionRequest:
    """
    Construct ChatCompletionRequest expected by vLLM >= 0.4,
    where messages must be a list[dict] / typed-dicts.
    """
    messages: List[dict] = [make_msg("system", system_prompt)]

    if body_type == "messages":          # already list[dict] from the dataset
        messages.extend(body)
    elif body_type == "prompt":
        messages.append(make_msg("user", body))
    else:
        raise ValueError(f"Unknown body_type: {body_type}")

    return ChatCompletionRequest(
        model=MODEL_NAME,
        messages=messages,
        max_tokens=SAMPLING_PARAMS.max_tokens,
        temperature=SAMPLING_PARAMS.temperature,
        top_p=SAMPLING_PARAMS.top_p,
        stop=SAMPLING_PARAMS.stop,
        stream=False,
    )


# --------------------------- PARSING  -------------------------------------- #
def parse_output(text: str) -> Dict[str, Any]:
    """
    Replace with your project-specific logic.
    Must return a dict of scalar numbers that are aggregatable by `sum`.
    Below: trivial example that measures length of the answer.
    """
    think_in_text = "</think>" in text
    return {
        "closed_thinking_tag": 1 if think_in_text else 0,
        "contains_thoughts": 1 if think_in_text and len(text.split("</think>")[0]) > 0 else 0
    }


# --------------------------- MAIN SCRIPT ----------------------------------- #
def main(args: argparse.Namespace) -> None:

    t0 = time.time()
    set_seed()

    # 1. Load 2×10k rows
    sampled_sets = {
        tag: load_and_sample(ds_name, col, NUM_SAMPLES)
        for tag, (ds_name, col) in DATASETS.items()
    }

    # 2. Instantiate vLLM
    llm = LLM(
        model=MODEL_NAME,
        revision=MODEL_REVISION,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_mem_util,
    )

    # 3. Build one big list of requests (each example used twice – 2 system prompts)
    requests: List[ChatCompletionRequest] = []
    meta_info = []           # (dataset_tag, system_idx) parallel list

    for sys_idx, sys_prompt in enumerate(SYSTEM_PROMPTS):
        for ds_tag, ds in sampled_sets.items():
            col_name = DATASETS[ds_tag][1]
            for row in ds:
                body = row[col_name]
                req = build_chat_request(sys_prompt, body, col_name)
                requests.append(req)
                meta_info.append((ds_tag, sys_idx))

    print(f"Total generations to run: {len(requests):,}")

    # 4. Run inference in batches
    generations: List[str] = [None] * len(requests)
    for start in tqdm(range(0, len(requests), BATCH_SIZE), desc="vLLM inference"):
        batch = requests[start:start+BATCH_SIZE]
        outs = llm.generate(batch, SAMPLING_PARAMS)
        for out in outs:
            # The response for each batch element is in out.outputs[0].text
            generations[start + out.index] = out.outputs[0].text

    assert None not in generations, "Some generations missing – indexing error?"

    # 5. Parse & aggregate
    aggregates: Dict[str, Dict[str, float]] = {}    # dataset -> sys_idx -> metric->value
    counts: Dict[str, Dict[str, int]] = {}

    for (ds_tag, sys_idx), text in zip(meta_info, generations):
        metrics = parse_output(text)

        key = f"{ds_tag}|sys{sys_idx}"
        if key not in aggregates:
            aggregates[key] = {k: 0.0 for k in metrics}
            counts[key] = 0

        for k, v in metrics.items():
            aggregates[key][k] += v
        counts[key] += 1

    # 6. Report
    print("\n=== Aggregate results ===")
    for ds_tag in sampled_sets.keys():
        for sys_idx, sys_prompt in enumerate(SYSTEM_PROMPTS):
            key = f"{ds_tag}|sys{sys_idx}"
            n = counts[key]
            print(f"\nDataset: {ds_tag:25} | System prompt #{sys_idx}:")
            for metric, total in aggregates[key].items():
                print(f"  {metric:20}: {total/n:.4f} (avg over {n})")

    print(f"\nFinished in {(time.time()-t0)/60:.1f} minutes")


# --------------------------- CLI ------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, default=1,
                        help="Tensor parallelism for vLLM.")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16"],
                        help="Model dtype.")
    parser.add_argument("--gpu-mem-util", type=float, default=0.90,
                        help="vLLM GPU memory utilisation fraction.")
    args = parser.parse_args()
    main(args)