"""
Naive simpleQA RAG baseline: query using the question, and then generate the answer.
python open_instruct/search_utils/naive_rag_baseline_eval.py \
    --dataset_name simpleqa \
    --model_path /path/to/model \
    --output_dir tmp
"""

import argparse
import json
import os

import ray
import vllm
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.ground_truth_utils import f1_score
from open_instruct.search_utils.massive_ds import get_snippets_for_query

ray.init()

PROMPT = "Answer the given question. You must conduct reasoning "
"inside <think> and </think> first every time you get new information. "
"After reasoning, if you find you lack some knowledge, you can call a "
"search engine by <query> query </query>, and it will return the top "
"searched results between <document> and </document>. You can search as "
"many times as you want. If you find no further external knowledge needed, "
"you can directly provide the answer inside <finish> and </finish> without "
"detailed illustrations. For example, <finish> xxx </finish>. Question: "


parser = argparse.ArgumentParser(description="Eval SimpleQA using the search actor.")
parser.add_argument(
    "--dataset_name", type=str, choices=["hotpotqa", "nq", "tqa", "2wiki", "simpleqa"], help="Dataset name."
)
parser.add_argument("--model_path", type=str, help="Path to the model.")
parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
parser.add_argument("--output_dir", type=str, default="tmp", help="Output directory.")
parser.add_argument("--max_eval_samples", type=int, default=2000, help="Max eval samples.")
parser.add_argument(
    "--add_finish", action="store_true", help="Whether to forcibly add the finish tag to the end of the prompt."
)
args = parser.parse_args()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

model = vllm.LLM(model=args.model_path, revision=args.model_revision, tokenizer_revision=args.tokenizer_revision)

# always use no prompt and just search.
if args.dataset_name == "simpleqa":
    ds = load_dataset("hamishivi/SimpleQA-RLVR-noprompt", split="test")
else:
    ds = load_dataset(f"rulins/{args.dataset_name}_rlvr_no_prompt", split="test")

if args.max_eval_samples > -1 and args.max_eval_samples < len(ds):
    ds = ds.shuffle(42).select(range(args.max_eval_samples))

queries = [x[0]["content"].strip() for x in ds["messages"]]
# manually do the search
query_results = [get_snippets_for_query(query)[0] for query in queries]

prompt_strings = []
for i, sample in enumerate(ds):
    msgs = sample["messages"]
    msgs[0]["content"] = PROMPT + msgs[0]["content"]
    s = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    prompt_strings.append(s)

# add the snippets and query in
for i, prompt in enumerate(prompt_strings):
    # add the snippets and query in
    snippets = query_results[i]
    query = queries[i]
    # add the query and snippets to the prompt
    prompt_strings[i] = prompt + f"<query>{query}</query><document>{snippets}</document>"
    # manually force the finish tag to be added.
    if args.add_finish:
        prompt_strings[i] += "<finish>"

# use greedy decoding
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=args.model_len,
    include_stop_str_in_output=True,
    n=1,
    stop=["</query>", "</finish>"],  # needed for search actor (TODO: make this api nicer)
)

result = model.generate(prompt_strings, sampling_params=sampling_params)
# grab text answers
generations = [x.outputs[0].text for x in result]
# parse out answer
if not args.add_finish:
    predictions = [x.split("<finish>")[-1].split("</finish>")[0].lower() for x in generations]
else:
    # if we manually added the finish tag, just assume the answer is the whole generation.
    predictions = [x.replace("<finish>", "").replace("</finish>", "") for x in generations]
labels = [data["ground_truth"].lower() for data in ds]
# calculate string f1
f1_scores = [f1_score(predictions[i], labels[i]) for i in range(len(predictions))]
f1s = [x["f1"] for x in f1_scores]
recalls = [x["recall"] for x in f1_scores]
precisions = [x["precision"] for x in f1_scores]
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
