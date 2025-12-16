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

import argparse
import json
import os

import numpy as np
import pandas as pd
import tqdm
from datasets import load_dataset
from huggingface_hub import repo_exists
from transformers import AutoTokenizer

import open_instruct.utils as open_instruct_utils


def get_statistics_for_messages_data(
    data_path, dataset=None, split="train", messages_key="messages", tokenizer="philschmid/meta-llama-3-tokenizer"
):
    if dataset is None:
        # load dataset
        dataset = load_dataset("json", data_files={split: data_path}, num_proc=open_instruct_utils.max_num_processes())
    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
    # get statistics
    num_instances = len(dataset[split])

    # remove any messages that have "role" == "system"
    def remove_system_messages(example):
        example[messages_key] = [message for message in example[messages_key] if message["role"] != "system"]
        return example

    dataset = dataset.map(remove_system_messages, num_proc=16)

    num_of_turns = [len(instance[messages_key]) for instance in dataset[split]]
    user_prompt_lengths = []
    assistant_response_lengths = []
    instance_lengths = []
    for instance in tqdm.tqdm(dataset[split], desc="Processing instances"):
        instance_length = 0
        for message in instance[messages_key]:
            if message["role"] == "user":
                user_prompt_lengths.append(
                    len(tokenizer(message["content"], truncation=False, add_special_tokens=False)["input_ids"])
                )
                instance_length += user_prompt_lengths[-1]
            elif message["role"] == "assistant":
                assistant_response_lengths.append(
                    len(tokenizer(message["content"], truncation=False, add_special_tokens=False)["input_ids"])
                )
                instance_length += assistant_response_lengths[-1]
        instance_lengths.append(instance_length)

    top_100_longest_instances = np.argsort(instance_lengths)[-100:][::-1].tolist()
    if "id" in dataset[split].features:
        top_100_longest_instances = [dataset[split][i]["id"] for i in top_100_longest_instances]
    else:
        top_100_longest_instances = None

    result = {
        "num_instances": num_instances,
        "turns_summary": pd.Series(num_of_turns).describe(),
        "user_prompt_lengths_summary": pd.Series(user_prompt_lengths).describe(),
        "assistant_response_lengths_summary": pd.Series(assistant_response_lengths).describe(),
        "total_lengths_summary": pd.Series(instance_lengths).describe(),
        "num_instances_with_total_length_gt_512": np.sum(np.array(instance_lengths) > 512),
        "num_instances_with_total_length_gt_768": np.sum(np.array(instance_lengths) > 768),
        "num_instances_with_total_length_gt_1024": np.sum(np.array(instance_lengths) > 1024),
        "num_instances_with_total_length_gt_1536": np.sum(np.array(instance_lengths) > 1536),
        "num_instances_with_total_length_gt_2048": np.sum(np.array(instance_lengths) > 2048),
        "num_instances_with_total_length_gt_4096": np.sum(np.array(instance_lengths) > 4096),
        "top_100_longest_instances": top_100_longest_instances,
    }

    # convert everything to dict or scalar
    for key, value in result.items():
        if isinstance(value, pd.Series):
            result[key] = value.to_dict()
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.int64):
            result[key] = int(value)

    return result


def get_statistics_for_prompt_completion_data(
    data_path, dataset=None, split="train", response_key="completion", tokenizer="philschmid/meta-llama-3-tokenizer"
):
    if dataset is None:
        # load dataset
        dataset = load_dataset("json", data_files={split: data_path}, num_proc=open_instruct_utils.max_num_processes())
    prompts = [instance["prompt"] for instance in dataset[split]]
    completions = [instance[response_key] for instance in dataset[split]]
    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenized_prompts = tokenizer(prompts, truncation=False, add_special_tokens=False)
    tokenized_completions = tokenizer(completions, truncation=False, add_special_tokens=False)
    # get statistics
    num_instances = len(dataset[split])
    prompt_lengths = [len(tokenized_prompts["input_ids"][i]) for i in range(num_instances)]
    completion_lengths = [len(tokenized_completions["input_ids"][i]) for i in range(num_instances)]
    prompt_completion_lengths = [prompt_lengths[i] + completion_lengths[i] for i in range(num_instances)]

    result = {
        "num_instances": num_instances,
        "prompt_lengths_summary": pd.Series(prompt_lengths).describe(),
        "completion_lengths_summary": pd.Series(completion_lengths).describe(),
        "prompt_completion_lengths_summary": pd.Series(prompt_completion_lengths).describe(),
        "num_instances_with_prompt_length_gt_512": np.sum(np.array(prompt_lengths) > 512),
        "num_instances_with_completion_length_gt_512": np.sum(np.array(completion_lengths) > 512),
        "num_instances_with_prompt_completion_length_gt_512": np.sum(np.array(prompt_completion_lengths) > 512),
        "num_instances_with_completion_length_gt_768": np.sum(np.array(completion_lengths) > 768),
        "num_instances_with_prompt_completion_length_gt_1024": np.sum(np.array(prompt_completion_lengths) > 1024),
    }

    # convert everything to dict or scalar
    for key, value in result.items():
        if isinstance(value, pd.Series):
            result[key] = value.to_dict()
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.int64):
            result[key] = int(value)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, help="Path to save the statistics.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--response_key", type=str, default="completion")
    parser.add_argument("--messages_key", type=str, default="messages")
    parser.add_argument("--tokenizer", type=str, default="philschmid/meta-llama-3-tokenizer")
    args = parser.parse_args()

    # Check if the data_path is a dataset id, only check if /
    if "json" in args.data_path:
        with open(args.data_path) as f:
            sample = json.loads(f.readline())
        dataset = None

    elif repo_exists(args.data_path, repo_type="dataset"):
        dataset = load_dataset(args.data_path, num_proc=open_instruct_utils.max_num_processes())
        sample = dataset[args.split][0]
    else:
        raise ValueError("Invalid data path - the data path should be either a dataset id or a path to a json file.")

    if args.messages_key in sample:
        statistics = get_statistics_for_messages_data(
            args.data_path, dataset=dataset, split=args.split, messages_key=args.messages_key, tokenizer=args.tokenizer
        )
    elif "prompt" in sample:
        statistics = get_statistics_for_prompt_completion_data(
            args.data_path, dataset=dataset, split=args.split, response_key=args.response_key, tokenizer=args.tokenizer
        )
    else:
        raise ValueError("Invalid data format - the data should be either prompt completion data or messages data.")

    print(json.dumps(statistics, indent=4))

    if args.save_path is not None:
        # if save path doesn't exist, make it
        if not os.path.exists(os.path.dirname(args.save_path)):
            os.makedirs(os.path.dirname(args.save_path))
        with open(args.save_path, "w") as f:
            json.dump(statistics, f, indent=4)
            print(f"Statistics saved to {args.save_path}")
