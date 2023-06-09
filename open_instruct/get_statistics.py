import json
import os
import sys
import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer


def get_statistics_for_messages_data(data_path):
    # load dataset
    dataset = load_dataset("json", data_files={"train": data_path})
    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B", use_fast=False)
    # get statistics
    num_instances = len(dataset["train"])
    num_of_turns = [len(instance["messages"]) for instance in dataset["train"]]
    user_prompt_lengths = []
    assistant_response_lengths = []
    instance_lengths = []
    for instance in tqdm.tqdm(dataset["train"], desc="Processing instances"):
        instance_length = 0
        for message in instance["messages"]:
            if message["role"] == "user":
                user_prompt_lengths.append(len(tokenizer(message["content"], truncation=False, add_special_tokens=False)["input_ids"]))
                instance_length += user_prompt_lengths[-1]
            elif message["role"] == "assistant":
                assistant_response_lengths.append(len(tokenizer(message["content"], truncation=False, add_special_tokens=False)["input_ids"]))
                instance_length += assistant_response_lengths[-1]
        instance_lengths.append(instance_length)

    top_100_longest_instances = np.argsort(instance_lengths)[-100:][::-1].tolist()
    top_100_longest_instances = [dataset["train"][i]["id"] for i in top_100_longest_instances]

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

def get_statistics_for_prompt_completion_data(data_path):
    # load dataset
    dataset = load_dataset("json", data_files={"train": data_path})
    prompts = [instance["prompt"] for instance in dataset["train"]]
    completions = [instance["completion"] for instance in dataset["train"]]
    # tokenize dataset
    tokenizer = AutoTokenizer.from_pretrained("/net/nfs.cirrascale/allennlp/yizhongw/hf_llama_models/7B")
    tokenized_prompts = tokenizer(prompts, truncation=False, add_special_tokens=False)
    tokenized_completions = tokenizer(completions, truncation=False, add_special_tokens=False)
    # get statistics
    num_instances = len(dataset["train"])
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
    args = parser.parse_args()
    
    with open(args.data_path, "r") as f:
        sample = json.loads(f.readline())
    if "prompt" in sample:
        statistics = get_statistics_for_prompt_completion_data(args.data_path)
    elif "messages" in sample:
        statistics = get_statistics_for_messages_data(args.data_path)
    else:
        raise ValueError("Invalid data format - the data should be either prompt completion data or messages data.")

    print(json.dumps(statistics, indent=4))

    if args.save_path is not None:
        with open(args.save_path, "w") as f:
            json.dump(statistics, f, indent=4)