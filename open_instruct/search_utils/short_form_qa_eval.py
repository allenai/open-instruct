'''
Eval short-form QA using the search actor.
'''
import os
import argparse
import ray
import json
from vllm import SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from open_instruct.search_utils.search_actor import LLMSearchRayActor
from open_instruct.ground_truth_utils import f1_score
from open_instruct.vllm_utils2 import ray_noset_visible_devices
ray.init()

parser = argparse.ArgumentParser(description="Eval SimpleQA using the search actor.")
parser.add_argument("--dataset_name", type=str, choices=["hotpotqa", "nq", "tqa", "2wiki", "simpleqa"], help="Dataset name.")
parser.add_argument("--no_prompt", action="store_true", help="Whether to use no prompt.")
parser.add_argument("--model_path", type=str, help="Path to the model.")
parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
parser.add_argument("--max_eval_samples", type=int, default=2000, help="Max eval samples.")
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
if args.dataset_name == "simpleqa":
    if args.no_prompt:
        ds = load_dataset("hamishivi/SimpleQA-RLVR-noprompt", split="test")
    else:
        ds = load_dataset("hamishivi/SimpleQA-RLVR", split="test")
else:
    ds = load_dataset(f"hamishivi/{args.dataset_name}_rlvr{'_no_prompt' if args.no_prompt else ''}", split="test")
prompt_token_ids = [tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True) for data in ds]

if args.max_eval_samples > -1:
    ds = ds.shuffle(42).select(range(args.max_eval_samples))

# use greedy decoding
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=args.model_len,
    include_stop_str_in_output=True,
    n=1,
    stop=["</query>", "</finish>"],  # needed for search actor (TODO: make this api nicer)
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

# Check if labels are JSON strings that need to be parsed
try:
    json.loads(labels[0])
    # If we get here, the labels are JSON strings
    labels = [json.loads(label) for label in labels]
except json.JSONDecodeError:
    # Labels are already plain strings, no processing needed
    labels = [[label] for label in labels]

# calculate string f1
f1_scores = [max([f1_score(predictions[i], label) for label in labels[i]], key=lambda x: x['f1']) for i in range(len(predictions))]
f1s = [x['f1'] for x in f1_scores]
recalls = [x['recall'] for x in f1_scores]
precisions = [x['precision'] for x in f1_scores]
avg_f1 = sum(f1s) / len(f1s)
print(f"Average F1: {avg_f1}")
avg_recall = sum(recalls) / len(recalls)
print(f"Average Recall: {avg_recall}")
avg_precision = sum(precisions) / len(precisions)
print(f"Average Precision: {avg_precision}")
# save predictions with sample data.
os.makedirs(args.output_dir, exist_ok=True)
with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
    for sample, prediction, generation in zip(ds, predictions, generations):
        f.write(json.dumps({**sample, "prediction": prediction, "generation": generation}) + "\n")
