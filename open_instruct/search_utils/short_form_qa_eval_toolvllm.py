"""
Eval short-form QA using the search actor.
"""

import argparse
import json
import os
import re

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.ground_truth_utils import f1_score
from open_instruct.search_utils.search_tool import SearchTool
from open_instruct.tool_utils.tool_vllm import ToolUseLLM

ray.init()

parser = argparse.ArgumentParser(description="Eval SimpleQA using the search actor.")
parser.add_argument(
    "--dataset_name", type=str, choices=["hotpotqa", "nq", "tqa", "2wiki", "simpleqa"], help="Dataset name."
)
parser.add_argument("--no_prompt", action="store_true", help="Whether to use no prompt.")
parser.add_argument("--model_path", type=str, help="Path to the model.")
parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
parser.add_argument("--max_eval_samples", type=int, default=2000, help="Max eval samples.")
parser.add_argument("--num_docs", type=int, default=3, help="Number of documents to retrieve.")
parser.add_argument(
    "--analyse_existing",
    type=str,
    default=None,
    help="Path to existing predictions to analyze. If specified, will not run the model again.",
)
parser.add_argument("--search_api_endpoint", type=str, default="http://localhost:8000", help="Search API endpoint.")
args = parser.parse_args()

# make output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

# load the GPQA test subsplit (gpqa diamond).
if args.dataset_name == "simpleqa":
    if args.no_prompt:
        ds = load_dataset("hamishivi/SimpleQA-RLVR-noprompt", split="test")
    else:
        ds = load_dataset("hamishivi/SimpleQA-RLVR", split="test")
else:
    ds = load_dataset(
        f"{'rulins' if args.no_prompt else 'hamishivi'}/{args.dataset_name}_rlvr{'_no_prompt' if args.no_prompt else ''}",
        split="test",
    )

if args.max_eval_samples > -1 and args.max_eval_samples < len(ds):
    ds = ds.shuffle(42).select(range(args.max_eval_samples))

prompt_token_ids = [tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True) for data in ds]

tool = SearchTool(
    start_str="<query>",
    end_str="</query>",
    api_endpoint=args.search_api_endpoint,
    number_documents_to_search=args.num_docs,
)

if not args.analyse_existing:
    # make actor.
    actor = ToolUseLLM(
        model=args.model_path,
        revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        tools={tool.end_str: tool},
        max_tool_calls=3,
        max_model_len=args.model_len,
    )
    # use greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.model_len,
        include_stop_str_in_output=True,
        n=1,
        stop=[tool.end_str, "</finish>"],
    )
    # Generate output using the actor.
    result = actor.generate(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
    # grab text answers - for tool vllm, we have to decode the output ids.
    generations = [x.outputs[0].token_ids for x in result]
    generations = [tokenizer.decode(x, skip_special_tokens=True) for x in generations]
    # parse out answer
    predictions = [x.split("<finish>")[-1].split("</finish>")[0].lower() for x in generations]
    labels = [data["ground_truth"].lower() for data in ds]
else:
    # Load existing predictions
    with open(args.analyse_existing, "r") as f:
        lines = f.readlines()
    generations = []
    predictions = []
    labels = [d["ground_truth"].lower() for d in ds]
    for line in lines:
        data = json.loads(line)
        generations.append(data["generation"])
        predictions.append(data["prediction"])

# Check if labels are JSON strings that need to be parsed
try:
    json.loads(labels[0])
    # If we get here, the labels are JSON strings
    labels = [json.loads(label) for label in labels]
except json.JSONDecodeError:
    # Labels are already plain strings, no processing needed
    labels = [[label] for label in labels]

# calculate string f1
f1_scores = [
    max([f1_score(predictions[i], label) for label in labels[i]], key=lambda x: x["f1"])
    for i in range(len(predictions))
]
f1s = [x["f1"] for x in f1_scores]
recalls = [x["recall"] for x in f1_scores]
precisions = [x["precision"] for x in f1_scores]
avg_f1 = sum(f1s) / len(f1s)
print(f"Average F1: {avg_f1}")
avg_recall = sum(recalls) / len(recalls)
print(f"Average Recall: {avg_recall}")
avg_precision = sum(precisions) / len(precisions)
print(f"Average Precision: {avg_precision}")

# some additional useful analyses:
# 1. how many predictions actually finished?
finished = [1 if x.lower().endswith("</finish>") else 0 for x in generations]
print(f"Finished: {sum(finished) / len(finished)}")
# 2. Of the predictions that finished, what is the f1 score?
f1_finished = [
    max([f1_score(predictions[i], label) for label in labels[i]], key=lambda x: x["f1"])
    for i in range(len(predictions))
    if finished[i]
]
f1s_finished = [x["f1"] for x in f1_finished]
avg_f1_finished = sum(f1s_finished) / len(f1s_finished)
print(f"Average F1 (finished only): {avg_f1_finished}")
# 3. How many predictions searched?
query_regex = r"<query>(.*?)</query>"
searched = [1 if re.search(query_regex, x) else 0 for x in generations]
print(f"Sent a query: {sum(searched) / len(searched)}")
# 3. Of the predictions that searched, what is the f1 score?
f1_searched = [
    max([f1_score(predictions[i], label) for label in labels[i]], key=lambda x: x["f1"])
    for i in range(len(predictions))
    if searched[i]
]
f1s_searched = [x["f1"] for x in f1_searched]
avg_f1_searched = sum(f1s_searched) / len(f1s_searched)
print(f"Average F1 (searched only): {avg_f1_searched}")
# What is the average number of times we search?
searches = [len(re.findall(query_regex, x)) for x in generations]
avg_searches = sum(searches) / len(searches)
print(f"Average no. searches: {avg_searches}")

# save predictions with sample data.
os.makedirs(args.output_dir, exist_ok=True)
with open(f"{args.output_dir}/predictions.jsonl", "w") as f:
    for sample, prediction, generation in zip(ds, predictions, generations):
        f.write(json.dumps({**sample, "prediction": prediction, "generation": generation}) + "\n")
