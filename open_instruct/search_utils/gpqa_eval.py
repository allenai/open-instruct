"""
Eval GPQA using the search actor.
"""

import argparse
import json
import os

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.search_utils.search_actor import LLMSearchRayActor
from open_instruct.vllm_utils3 import ray_noset_visible_devices

ray.init()

parser = argparse.ArgumentParser(description="Eval GPQA using the search actor.")
parser.add_argument("--model_path", type=str, help="Path to the model.")
parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
args = parser.parse_args()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

actor = LLMSearchRayActor.options(
    num_cpus=4,
    num_gpus=1,
    # VLLM v1 multiprocessing is required due to https://github.com/vllm-project/vllm/issues/15349
    runtime_env=ray.runtime_env.RuntimeEnv(env_vars={"VLLM_ENABLE_V1_MULTIPROCESSING": "0"}),
).remote(
    model=args.model_path,
    revision=args.model_revision,
    tokenizer_revision=args.tokenizer_revision,
    worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap",
    distributed_executor_backend="uni",
    bundle_indices=None,
    trust_remote_code=True,
    tensor_parallel_size=1,
    enforce_eager=True,
    dtype="bfloat16",
    seed=42,
    enable_prefix_caching=True,
    max_output_len=args.model_len,  # Explicitly set a custom max context length
    gpu_memory_utilization=0.95,
    num_gpus=1,
    enable_sleep_mode=False,
    noset_visible_devices=ray_noset_visible_devices(),
)

# load the GPQA test subsplit (gpqa diamond).
ds = load_dataset("hamishivi/GPQA-RLVR", split="test")
prompt_token_ids = [tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True) for data in ds]

# use greedy decoding
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=args.model_len,
    include_stop_str_in_output=True,
    n=1,
    stop=["</query>", "</finish>"],
)

# Generate output using the actor.
result = ray.get(
    actor.generate.remote(
        sampling_params=sampling_params,  # Uses default dummy sampling params.
        prompt_token_ids=prompt_token_ids,
        use_tqdm=False,
    )
)
# grab text answers
generations = [x.outputs[0].text for x in result]
# parse out answer
predictions = [x.split("<finish>")[-1].split("</finish>")[0].lower() for x in generations]
labels = [data["ground_truth"].lower() for data in ds]
# calculate accuracy
accuracy = sum([1 if predictions[i] == labels[i] else 0 for i in range(len(predictions))]) / len(predictions)
print(f"Accuracy: {accuracy}")
# save predictions with sample data.
os.makedirs(args.output_dir, exist_ok=True)
with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
    for sample, prediction, generation in zip(ds, predictions, generations):
        f.write(json.dumps({**sample, "prediction": prediction, "generation": generation}) + "\n")
