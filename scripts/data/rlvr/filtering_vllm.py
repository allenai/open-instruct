"""
python mason.py \
  --cluster ai2/jupiter --image nathanl/open_instruct_auto \
  --workspace ai2/tulu-thinker \
  --priority high \
  --preemptible \
  --gpus 1 \
  --num_nodes 1 \
  --max_retries 0 \
  -- python scripts/data/rlvr/filtering_vllm.py \
  --model hamishivi/qwen2_5_openthoughts2 \
  --dataset hamishivi/rlvr_orz_math_57k_collected \
  --split train \
  --offset 0 \
  --size 100000 \
  --output-file filtered_datasets/qwen2_5_openthoughts2/orz.jsonl \
  --number_samples 8
"""

import argparse
import json

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

import open_instruct.utils as open_instruct_utils
from open_instruct.dataset_transformation import CHAT_TEMPLATES


def main():
    parser = argparse.ArgumentParser(description="Bulk-generate N samples per HF dataset record using vLLM.")
    parser.add_argument("--model", required=True, help="vLLM model ID (e.g. facebook/opt-125m)")
    parser.add_argument("--dataset", required=True, help="HF dataset name (e.g. squad)")
    parser.add_argument("--split", default="train", help="Which split to load")
    parser.add_argument("--offset", type=int, required=True, help="Start index into the split")
    parser.add_argument("--size", type=int, required=True, help="Number of records to process")
    parser.add_argument("--output-file", default=None, help="Path for output JSONL")
    parser.add_argument(
        "--push_to_hub", default=None, type=str, help="Give a dataset name to push this data to the hub."
    )
    parser.add_argument("--chat_template", type=str, default=None, help="Chat template name")
    parser.add_argument("--number_samples", type=int, default=8, help="Number of samples to generate per record")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    args = parser.parse_args()

    # 1. Load and slice dataset
    ds = load_dataset(args.dataset, split=args.split, num_proc=open_instruct_utils.max_num_processes())
    ds = ds.shuffle(seed=42)  # so we dont just take first n samples
    up_to = min(args.offset + args.size, len(ds))
    subset = ds.select(range(args.offset, up_to))

    # 2. Tokenizer for chat templating
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if args.chat_template is not None:
        if args.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"Unknown chat template: {args.chat_template}")
        tokenizer.chat_template = CHAT_TEMPLATES[args.chat_template]

    # 3. Build prompts
    prompts = [
        tokenizer.apply_chat_template(
            sample["messages"][:-1] if len(sample["messages"]) > 1 else sample["messages"],
            add_generation_prompt=True,
            tokenize=False,
        )
        for sample in subset
    ]
    # 4. vLLM bulk generate
    llm = LLM(model=args.model, dtype="bfloat16", enable_prefix_caching=True)
    sampling_params = SamplingParams(temperature=args.temperature, n=args.number_samples, max_tokens=32768)
    outputs = llm.generate(prompts, sampling_params)

    # 5. Write out JSONL
    if args.output_file is not None:
        with open(args.output_file, "w", encoding="utf-8") as out_f:
            for sample, req_out in zip(subset, outputs):
                gen_texts = [o.text for o in req_out.outputs]
                enriched = dict(sample)
                enriched["output"] = gen_texts
                out_f.write(json.dumps(enriched, ensure_ascii=False) + "\n")

    if args.push_to_hub is not None:
        dataset = load_dataset(args.dataset, split=args.split, num_proc=open_instruct_utils.max_num_processes())
        dataset.push_to_hub(args.push_to_hub)


if __name__ == "__main__":
    main()
