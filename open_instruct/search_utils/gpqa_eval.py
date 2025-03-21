'''
Eval GPQA using the search actor.
'''
import argparse
import ray
import json
from vllm import SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer
from open_instruct.search_utils.search_actor import LLMSearchRayActor
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
    num_gpus=1
).remote(
    args.model_path,
    revision=args.model_revision,
    tokenizer_revision=args.tokenizer_revision,
    trust_remote_code=True,
    tensor_parallel_size=1,
    enforce_eager=True,
    dtype="bfloat16",
    seed=42,
    enable_prefix_caching=True,
    max_model_len=args.model_len,  # This will be used as default max_context_length if not explicitly set
    max_context_length=args.model_len,  # Explicitly set a custom max context length
    gpu_memory_utilization=0.95,
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
    stop=["</query>"],  # needed for search actor (TODO: make this api nicer)
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
predictions = [x[0].text for x in result]
# parse out answer
predictions = [x.split("<answer>")[-1].split("</answer>")[0].lower() for x in predictions]
labels = [data["ground_truth"].lower() for data in ds]
# calculate accuracy
accuracy = sum([1 if predictions[i] == labels[i] else 0 for i in range(len(predictions))]) / len(predictions)
print(f"Accuracy: {accuracy}")
# save predictions with sample data.
with open(f"{args.output_dir}/predictions.txt", "w") as f:
    for sample, prediction in zip(ds, predictions):
        f.write(json.dumps({**ds, "prediction": prediction}) + "\n")
