import os
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

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
    model_name_or_path: str
    output_path: Optional[str] = None
    model_revision: Optional[str] = None
    judge_model: str = "gpt-4o-2024-08-06"
    max_token: int = 1024
    n: int = 1000

    def __post_init__(self):
        if self.output_path is None:
            self.output_path = f"csvs/{self.model_name_or_path}_{self.model_revision}.csv"


def run_command(command: str):
    command_list = shlex.split(command)
    print(f"running {command}")
    subprocess.run(command_list, stderr=sys.stderr, stdout=sys.stdout)


def main(args: Args):
    directory = os.path.dirname(args.output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

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

    ds = raw_datasets.map(process, remove_columns=raw_datasets.column_names, num_proc=4)
    prompts = ds["prompt_str"]
    reference_summaries = ds["response_str"]

    judged_df_path = args.output_path.replace(".csv", "_judged.csv")
    if not os.path.exists(judged_df_path):

        sampling_params = SamplingParams(temperature=0.0, top_p=0.95, max_tokens=args.max_token)
        llm = LLM(
            model=args.model_name_or_path,
            revision=args.model_revision,
            tokenizer_revision=args.model_revision,
            tensor_parallel_size=1,
            max_model_len=4096,
            disable_async_output_proc=True,
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
    else:
        df = pd.read_csv(args.output_path)
        judged_df = pd.read_csv(judged_df_path)
    print(judged_df["preferred"].value_counts())
    print(judged_df["preferred"].value_counts(normalize=True))
    print(f"{df['model_response_len'].mean()=}")

    judged_df.to_csv(args.output_path.replace(".csv", "_judged.csv"))

    from open_instruct.hf_viz import print_hf_chosen_rejected
    for i in range(len(df)):
        reference_messages = raw_datasets[i]["messages"]
        model_messages = raw_datasets[i]["messages"][:-1] + [{"role": "assistant", "content": df["model_response"][i]}]
        chosen_messages = model_messages if judged_df["preferred"][i] == "response0" else reference_messages
        rejected_messages = reference_messages if judged_df["preferred"][i] == "response0" else model_messages
        message_source = "model response" if judged_df["preferred"][i] == "response0" else "reference response"
        print_hf_chosen_rejected(chosen_messages, rejected_messages, title=f"preferred: {message_source}", explanation=judged_df['explanation'][i])
        input("Press Enter to continue...")


if __name__ == "__main__":
    main(*HfArgumentParser((Args)).parse_args_into_dataclasses())