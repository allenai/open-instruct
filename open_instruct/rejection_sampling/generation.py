# Copyright 2024 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import multiprocessing
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from datasets import load_dataset
from rich.console import Console
from rich.pretty import pprint
from rich.table import Table
from transformers import AutoTokenizer, HfArgumentParser
from vllm import LLM, SamplingParams


@dataclass
class Args:
    model_name_or_path: str = "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr"
    save_filename: str = "completions.jsonl"


@dataclass
class GenerationArgs:
    n: int = 1
    """the number of samples to generate per prompt"""
    temperature: float = 0.8
    response_length: int = 53
    tensor_parallel_size: int = 1


@dataclass
class DatasetArgs:
    dataset_name: str = None
    dataset_text_field: str = "prompt"
    dataset_train_split: str = "train"
    dataset_test_split: str = "validation"
    dataset_start_idx: int = 0
    dataset_end_idx: Optional[int] = 100
    sanity_check: bool = False
    sanity_check_size: int = 100


def print_rich_table(df: pd.DataFrame) -> Table:
    console = Console()
    table = Table(show_lines=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row.astype(str).tolist())
    console.print(table)


def main(args: Args, dataset_args: DatasetArgs, gen_args: GenerationArgs):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    ds = load_dataset(dataset_args.dataset_name)
    if dataset_args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(min(dataset_args.sanity_check_size, len(ds[key]))))
    if dataset_args.dataset_end_idx is None:
        dataset_args.dataset_end_idx = len(ds[dataset_args.dataset_train_split])
    for key in ds:
        ds[key] = ds[key].select(range(dataset_args.dataset_start_idx, dataset_args.dataset_end_idx))
    pprint([dataset_args, args, gen_args])

    # DATASET specific logic: in this dataset the prompt is simply just a list of strings
    ds = ds.map(
        lambda x: {"prompt_token_ids": tokenizer.apply_chat_template(x["messages"][:-1])},
        num_proc=multiprocessing.cpu_count(),
    )
    prompt_token_ids = ds[dataset_args.dataset_train_split]["prompt_token_ids"]

    # Generate using vLLM
    llm = LLM(model=args.model_name_or_path, tensor_parallel_size=gen_args.tensor_parallel_size)
    outputs = llm.generate(
        prompt_token_ids=prompt_token_ids,
        sampling_params=SamplingParams(
            n=gen_args.n,
            temperature=gen_args.temperature,
            top_p=1.0,
            max_tokens=gen_args.response_length,
            include_stop_str_in_output=True,
        ),
    )
    # Assuming we generate n=3 completions per prompt, the outputs will look like:
    # prompt | completions
    # -------|------------
    # q1     | a1
    # q1     | a2
    # q1     | a3
    # q2     | a1
    # ...
    table = defaultdict(list)
    for output, messages in zip(outputs, ds[dataset_args.dataset_train_split]["messages"]):
        # if the model completions are exactly the same across all completions, we can skip this
        if len(set([item.text for item in output.outputs])) == 1:
            continue

        for item in output.outputs:
            new_messages = copy.deepcopy(messages[:-1])
            new_messages.append({"role": "assistant", "content": item.text})
            table["messages"].append(new_messages)
            table["model_completion"].append(item.text)
            table["reference_completion"].append(messages[-1]["content"])

    # print_rich_table(pd.DataFrame(table)) # uncomment this line to print the table

    # Save results
    os.makedirs(os.path.dirname(args.save_filename), exist_ok=True)
    with open(args.save_filename, "w") as outfile:
        for i in range(len(table["messages"])):
            json.dump(
                {
                    "messages": table["messages"][i],
                    "model_completion": table["model_completion"][i],
                    "reference_completion": table["reference_completion"][i],
                },
                outfile,
            )
            outfile.write("\n")


if __name__ == "__main__":
    parser = HfArgumentParser((Args, DatasetArgs, GenerationArgs))
    args, dataset_args, gen_args = parser.parse_args_into_dataclasses()
    main(args, dataset_args, gen_args)
