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


import asyncio
import copy
import json
import multiprocessing
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Optional

import pandas as pd
from api_generate import LLMGenerationConfig, LLMProcessor  # Import your classes
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
    skill: str = "chat"
    mode: str = "generation"  # Can be "generation" or "judgment"


@dataclass
class GenerationArgs:
    n: int = 1
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


async def generate_with_openai(model_name: str, data_list: list, args: Args, n: int):
    config = LLMGenerationConfig(model=model_name, n=n)
    processor = LLMProcessor(config)
    results = await processor.process_batch(data_list, args)
    return results


def generate_with_vllm(model_name_or_path: str, prompt_token_ids, gen_args: GenerationArgs):
    llm = LLM(model=model_name_or_path, tensor_parallel_size=gen_args.tensor_parallel_size)
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

    return [
        {
            "outputs": [asdict(out) for out in output.outputs],
            "prompt": output.prompt,
            "prompt_logprobs": output.prompt_logprobs,
            "metrics": output.metrics,
        }
        for output in outputs
    ]


def format_conversation(messages: list) -> str:
    formatted_conversation = []

    # Iterate through the messages
    for message in messages:  # Exclude the last assistant message
        role = "User A" if message["role"] == "user" else "User B"
        content = message["content"].strip()
        formatted_conversation.append(f"{role}: {content}")

    # Join the conversation with a single newline
    return "\n".join(formatted_conversation)


def main(args: Args, dataset_args: DatasetArgs, gen_args: GenerationArgs):

    ds = load_dataset(dataset_args.dataset_name)
    if dataset_args.sanity_check:
        for key in ds:
            ds[key] = ds[key].select(range(min(dataset_args.sanity_check_size, len(ds[key]))))
    if dataset_args.dataset_end_idx is None:
        dataset_args.dataset_end_idx = len(ds[dataset_args.dataset_train_split])
    for key in ds:
        ds[key] = ds[key].select(range(dataset_args.dataset_start_idx, dataset_args.dataset_end_idx))
    pprint([dataset_args, args, gen_args])

    if "gpt-3.5" in args.model_name_or_path or "gpt-4" in args.model_name_or_path:
        use_openai = True
    else:
        use_openai = False

    if use_openai:

        ds = ds.map(
            lambda x: {"prompt": format_conversation(x["messages"][:-1])},
            num_proc=multiprocessing.cpu_count(),
        )
        messages = ds[dataset_args.dataset_train_split]["prompt"]
        responses = asyncio.run(generate_with_openai(args.model_name_or_path, messages, args, gen_args.n))
        outputs = [{"outputs": [{"text": response}]} for response in responses]

    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

        ds = ds.map(
            lambda x: {"prompt_token_ids": tokenizer.apply_chat_template(x["messages"][:-1])},
            num_proc=multiprocessing.cpu_count(),
        )
        prompt_token_ids = ds[dataset_args.dataset_train_split]["prompt_token_ids"]
        # Generate using vLLM.
        outputs = generate_with_vllm(args.model_name_or_path, prompt_token_ids, gen_args)

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

        # if args.generation:
        #     # if the model completions are exactly the same across all completions per prompt, we can skip this
        #     if len(set(item["text"] for item in output["outputs"])) == 1:
        #         continue

        for item in output["outputs"]:
            new_messages = copy.deepcopy(messages[:-1])
            new_messages.append({"role": "assistant", "content": item["text"]})
            table["messages"].append(new_messages)
            table["model_completion"].append(item["text"])
            table["reference_completion"].append(messages[-1]["content"])

    # Save results
    os.makedirs(os.path.dirname(args.save_filename), exist_ok=True)
    with open(args.save_filename, "w") as outfile:
        for i in range(len(table["messages"])):
            json.dump(
                {
                    "messages": table["messages"][i],
                    "model_completion": table["model_completion"][i],
                    "reference_completion": table["reference_completion"][i],
                    "model": args.model_name_or_path,
                },
                outfile,
            )
            outfile.write("\n")


if __name__ == "__main__":
    parser = HfArgumentParser((Args, DatasetArgs, GenerationArgs))
    args, dataset_args, gen_args = parser.parse_args_into_dataclasses()
    main(args, dataset_args, gen_args)
