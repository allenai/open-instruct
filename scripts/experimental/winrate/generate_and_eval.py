import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd
from datasets import load_dataset
from gpt_eval_judge import LLMJudgeConfig, llm_judge
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams


"""
python -i winrate/generate_and_eval.py \
    --model_name_or_path ai2-adapt-dev/sft-norobot \
    --output_path test.csv \
    --n 1000
"""


@dataclass
class Args:
    output_path: str
    model_name_or_path: str
    model_revision: str = "main"
    judge_model: str = "gpt-4o-2024-08-06"
    n: int = 1000


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    subprocess.run(command_list, stderr=sys.stderr, stdout=sys.stdout)


MAX_TOKENS = 200  # a very generous max token length
parser = HfArgumentParser(Args)
args = parser.parse_args_into_dataclasses()[0]
tokenizer = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    revision=args.model_revision,
)
raw_datasets = load_dataset("HuggingFaceH4/no_robots", split="test")
def process(x):
    x["complete_messages_str"] = tokenizer.apply_chat_template(x["messages"], tokenize=False)
    x["prompt_str"] = tokenizer.apply_chat_template(x["messages"][:-1], tokenize=False, add_generation_prompt=True)
    response_str = x["complete_messages_str"]
    response_str = response_str[len(x["prompt_str"]) :]
    x["response_str"] = response_str.strip()
    return x

raw_datasets = raw_datasets.map(process, remove_columns=raw_datasets.column_names, num_proc=4)
prompts = raw_datasets["prompt_str"]
reference_summaries = raw_datasets["response_str"]

sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=MAX_TOKENS)
llm = LLM(
    model=args.model_name_or_path,
    revision=args.model_revision,
    tokenizer_revision=args.model_revision,
    tensor_parallel_size=1,
)
outputs = llm.generate(prompts, sampling_params)
table = defaultdict(list)

# Print the outputs.
for output, reference_response in zip(outputs, reference_summaries):
    prompt = output.prompt
    generated_text = output.outputs[0].text
    table["prompt"].append(prompt)
    table["model_response"].append(generated_text.strip())  # need `strip()` because of the leading space
    table["model_response_len"].append(len(output.outputs[0].token_ids))
    table["reference_response"].append(reference_response)
    table["reference_response_len"].append(
        len(tokenizer(f" {reference_response}")["input_ids"])
    )  # prepend leading space

df = pd.DataFrame(table)
df.to_csv(args.output_path)

#####
# GPT as a judge
####
df["response0"] = df["model_response"]
df["response1"] = df["reference_response"]
judged_df = llm_judge(
    LLMJudgeConfig(
        n=args.n,
        model=args.judge_model,
    ),
    df,
)
print(judged_df["preferred"].value_counts())
# print percentage
print(judged_df["preferred"].value_counts(normalize=True))

judged_df.to_csv(args.output_path.replace(".csv", "_judged.csv"))