"""
Eval short-form QA using the search actor.
example:
python -m open_instruct.search_utils.astabench_generate_toolvllm \
    --dataset_file rubrics_v1.json \
    --add_prompt \
    --model_path /weka/oe-adapt-default/hamishi/model_checkpoints/rl_rag/1006_rl_rag_open_scholar_ppo__1__1749687970_checkpoints/step_700 \
    --output_path astabench_ppo_generate_file.jsonl \
    --num_docs 3 \
    --search_api_endpoint http://root@saturn-cs-aus-232.reviz.ai2.in:47695/search
"""

import argparse
import json
import os
import re

import ray
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.search_utils.search_tool import SearchTool
from open_instruct.tool_utils.tool_vllm import ToolUseLLM

ray.init()

PROMPT = "Answer the given question. You must conduct reasoning inside <think> and </think> first every time you get new information. After reasoning, if you find you lack some knowledge, you can call a search engine by <query> query </query>, and it will return the top searched results between <document> and </document>. You can search as many times as you want. If you find no further external knowledge needed, you can directly provide the answer inside <finish> and </finish>. For example, <finish> xxx </finish>. Question: <question>"

parser = argparse.ArgumentParser(description="Eval SimpleQA using the search actor.")
parser.add_argument(
    "--dataset_file", type=str, help="Path to the dataset file."
)
parser.add_argument("--add_prompt", action="store_true", help="Whether to explicitly add a prompt or not.")
parser.add_argument("--model_path", type=str, help="Path to the model.")
parser.add_argument("--model_revision", type=str, default="main", help="Model revision.")
parser.add_argument("--tokenizer_revision", type=str, default="main", help="Tokenizer revision.")
parser.add_argument("--model_len", type=int, default=8192, help="Max model length.")
parser.add_argument("--output_path", type=str, default="tmp", help="Output path.")
parser.add_argument("--num_docs", type=int, default=3, help="Number of documents to retrieve.")
parser.add_argument("--search_api_endpoint", type=str, default="http://localhost:8000", help="Search API endpoint.")
args = parser.parse_args()

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.tokenizer_revision)

# load the dataset, assuming same format as:
# https://huggingface.co/datasets/allenai/asta-bench/blob/main/tasks/sqa/rubrics_v1.json
with open(args.dataset_file, "r") as f:
    ds = json.load(f)

prompt_token_ids = []
for data in ds:
    prompt = data['question']
    if args.add_prompt:
        prompt = PROMPT + "\n" + prompt
    prompt_token_ids.append(
        tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True)
    )

tool = SearchTool(
    start_str="<query>",
    end_str="</query>",
    api_endpoint=args.search_api_endpoint,
    number_documents_to_search=args.num_docs,
)

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
result = actor.generate(
    sampling_params=sampling_params,
    prompt_token_ids=prompt_token_ids,
)
# grab text answers - for tool vllm, we have to decode the output ids.
generations = [x.outputs[0].token_ids for x in result]
generations = [tokenizer.decode(x, skip_special_tokens=True) for x in generations]

# decompose into reasoning and answer
reasoning = []
answer = []
for generation in generations:
    reasoning = generation.split("</think>")[0]
    answer = generation.split("</think>")[-1]
    answer = answer.replace("<finish>", "").replace("</finish>", "")
    reasoning.append(reasoning)
    answer.append(answer)


# construct outputs. We need jsonl with question/answer
os.makedirs(args.output_path, exist_ok=True)
with open(f"{args.output_path}", "w") as f:
    for sample, reasoning, answer in zip(ds, reasoning, answer):
        f.write(json.dumps({**sample, "reasoning": reasoning, "answer": answer}) + "\n")
