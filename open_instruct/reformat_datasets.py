#!/usr/bin/env python
# coding=utf-8
'''
This script is used to reformat the downloaded datasets into the format that can be used by the model.
Here we use jsonl for the converted data. Each line in the jsonl file is a json object formatted as follows:
{
    "dataset": "dataset_name",
    "id": "unique_id",
    "messages": [
        {"role": "system", "content": "message_text"}, # optional
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        {"role": "user", "content": "message_text"},
        {"role": "assistant", "content": "message_text"},
        ...
    ],
}
'''

import json
import random
import re
import os
import argparse
from instruction_encode_templates import encode_instruction_example, encode_few_shot_example


def convert_super_ni_data(data_dir, output_dir, zero_shot_examples_per_task=60, few_shot_examples_per_task=20, n_few_shot=2):
    os.makedirs(output_dir, exist_ok=True)
    train_tasks = []
    with open(os.path.join(data_dir, "splits", "xlingual", "train_tasks.txt"), "r") as fin:
        for line in fin:
            if not "_mmmlu_" in line:   # skip mmlu to avoid test leakage
                train_tasks.append(line.strip())
    with open(os.path.join(output_dir, "super_ni_data.jsonl"), "w") as fout:
        for task in train_tasks:
            with open(os.path.join(data_dir, "tasks", f"{task}.json"), "r") as fin:
                task_data = json.load(fin)
            instruction = task_data["Definition"][0]
            if zero_shot_examples_per_task + few_shot_examples_per_task < len(task_data["Instances"]):
                instances = random.sample(task_data["Instances"], k=zero_shot_examples_per_task+few_shot_examples_per_task)
            else:
                instances = task_data["Instances"]
            for instance in instances[:zero_shot_examples_per_task]:
                encoded_example = encode_instruction_example(
                    instruction=instruction, 
                    input=instance["input"], 
                    output=instance["output"][0],
                    random_template=True,
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "super_ni",
                    "id": f"super_ni_{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
            for instance in instances[zero_shot_examples_per_task:]:
                if n_few_shot < len(task_data["Positive Examples"]):
                    examplars = random.sample(task_data["Positive Examples"], k=n_few_shot)
                else:
                    examplars = task_data["Positive Examples"]
                encoded_example = encode_few_shot_example(
                    instruction=instruction,
                    examplars=examplars,
                    input=instance["input"],
                    output=instance["output"][0],
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "super_ni",
                    "id": f"super_ni_{instance['id']}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
            
            
def convert_cot_data(data_dir, output_dir, num_zero_shot_examples=50000, num_few_shot_examples=50000):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if num_few_shot_examples > 0:
        with open(os.path.join(data_dir, "cot_zsopt.jsonl"), "r") as fin:
            zero_shot_examples = [json.loads(line) for line in fin]
            if num_zero_shot_examples < len(zero_shot_examples):
                zero_shot_examples = random.sample(zero_shot_examples, k=num_zero_shot_examples)
            examples.extend(zero_shot_examples)
    if num_few_shot_examples > 0:
        with open(os.path.join(data_dir, "cot_fsopt.jsonl"), "r") as fin:
            few_shot_examples = [json.loads(line) for line in fin]
            if num_few_shot_examples < len(few_shot_examples):
                few_shot_examples = random.sample(few_shot_examples, k=num_few_shot_examples)
            examples.extend(few_shot_examples)
    output_path = os.path.join(output_dir, "cot_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            fout.write(json.dumps({
                "dataset": "cot",
                "id": f"cot_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }) + "\n")
            

def convert_flan_v2_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "flan_v2_resampled_100k.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    output_path = os.path.join(output_dir, "flan_v2_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            prompt = example["inputs"]
            if not prompt.endswith("\n") and not prompt.rstrip().endswith(":"):
                prompt += "\n"
            completion = example["targets"]
            fout.write(json.dumps({
                "dataset": "flan_v2",
                "id": f"flan_v2_{idx}",
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]
            }) + "\n")


def convert_dolly_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "databricks-dolly-15k.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    output_path = os.path.join(output_dir, "dolly_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["context"], 
                output=example["response"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "dolly",
                "id": f"dolly_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_self_instruct_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "all_instances_82K.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    output_path = os.path.join(output_dir, "self_instruct_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "self_instruct",
                "id": f"self_instruct_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_unnatural_instructions_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    instance_cnt = 0
    with open(os.path.join(data_dir, "core_data.jsonl"), "r") as fin, open((os.path.join(output_dir, "unnatural_instructions_data.jsonl")), "w") as fout:
        for line in fin:
            task_data = json.loads(line)
            instruction = task_data["instruction"]
            for instance in task_data["instances"]:
                if instance["constraints"] and instance["constraints"].lower() not in ["none", "none."]:
                    instance_instruction = instruction + "\n" + instance["constraints"]
                else:
                    instance_instruction = instruction
                encoded_example = encode_instruction_example(
                    instruction=instance_instruction,
                    input=instance["input"],
                    output=instance["output"],
                    random_template=True,
                    eos_token=None
                )
                fout.write(json.dumps({
                    "dataset": "unnatural_instructions",
                    "id": f"unnatural_instructions_{instance_cnt}",
                    "messages": [
                        {"role": "user", "content": encoded_example["prompt"]},
                        {"role": "assistant", "content": encoded_example["completion"]},
                    ]
                }) + "\n")
                instance_cnt += 1


def convert_stanford_alpaca_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "alpaca_data.json"), "r") as fin:
        examples.extend(json.load(fin))
    output_path = os.path.join(output_dir, "stanford_alpaca_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "stanford_alpaca",
                "id": f"stanford_alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_code_alpaca_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "code_alpaca_20k.json"), "r") as fin:
        examples.extend(json.load(fin))
    output_path = os.path.join(output_dir, "code_alpaca_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "code_alpaca",
                "id": f"code_alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_gpt4_alpaca_data(data_dir, output_dir, load_en=True, load_zh=False):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if load_en:
        with open(os.path.join(data_dir, "alpaca_gpt4_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if load_zh:
        with open(os.path.join(data_dir, "alpaca_gpt4_data_zh.json"), "r") as fin:
            examples.extend(json.load(fin))
    output_path = os.path.join(output_dir, "gpt4_alpaca_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            encoded_example = encode_instruction_example(
                instruction=example["instruction"], 
                input=example["input"], 
                output=example["output"],
                random_template=True,
                eos_token=None
            )
            fout.write(json.dumps({
                "dataset": "gpt4_alpaca",
                "id": f"gpt4_alpaca_{idx}",
                "messages": [
                    {"role": "user", "content": encoded_example["prompt"]},
                    {"role": "assistant", "content": encoded_example["completion"]},
                ]
            }) + "\n")


def convert_sharegpt_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "sharegpt_html_cleaned_and_split.json"), "r") as fin:
        examples.extend(json.load(fin))

    output_path = os.path.join(output_dir, "sharegpt_data.jsonl")
    with open(output_path, "w") as fout:
        invalid_cnt = 0
        for idx, example in enumerate(examples):
            messages = []
            valid = True
            for message in example["conversations"]:
                if message["from"] == "human" or message["from"] == "user":
                    messages.append({
                        "role": "user",
                        "content": message["value"]
                    })
                elif message["from"] == "gpt" or message["from"] == "chatgpt":
                    messages.append({
                        "role": "assistant",
                        "content": message["value"]
                    })
                elif message["from"] == "system":
                    valid = False
                    invalid_cnt += 1
                    break
                elif message["from"] == "bing":
                    valid = False
                    invalid_cnt += 1
                    break
                else:
                    raise ValueError(f"Unknown message sender: {message['from']}")
            if messages and valid:
                fout.write(json.dumps({
                    "dataset": "sharegpt",
                    "id": f"sharegpt_{example['id']}",
                    "messages": messages
                }) + "\n")
        print(f"# of invalid examples in sharegpt data: {invalid_cnt}")


def convert_baize_data(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    for source in ["alpaca", "medical", "quora", "stackoverflow"]:
        with open(os.path.join(data_dir, f"{source}_chat_data.json"), "r") as fin:
            examples.extend(json.load(fin))

    output_path = os.path.join(output_dir, "baize_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            # split example["input"] by [|Human|] and [|AI|]
            messages = []
            rounds = example["input"].split("[|Human|]")[1:]
            for round in rounds:
                if not round.strip() or "[|AI|]" not in round:
                    continue
                human, assistant = round.split("[|AI|]")
                messages.append({
                    "role": "user",
                    "content": human.strip()
                })
                messages.append({
                    "role": "assistant",
                    "content": assistant.strip()
                })
            fout.write(json.dumps({
                "dataset": "baize",
                "id": f"baize_{idx}",
                "messages": messages
            }) + "\n")


def convert_oasst1_data(data_dir, output_dir):
    '''
    For OASST1, because it's in a tree structure, where every user input might get multiple replies, 
    we have to save every path from the root node to the assistant reply (including both leaf node and intemediate node).
    This results in some of the messages being duplicated among different paths (instances).
    Be careful when using this dataset for training. Ideally, you should only minimize the loss of the last message in each path.
    '''
    os.makedirs(output_dir, exist_ok=True)
    conversations = []
    with open(os.path.join(data_dir, "2023-04-12_oasst_ready.trees.jsonl"), "r") as fin:
        for line in fin:
            conversations.append(json.loads(line))

    output_path = os.path.join(output_dir, "oasst1_data.jsonl")

    # tranvers the conversation tree, and collect all valid sequences
    def dfs(reply, messages, valid_sequences):
        if reply["role"] == "assistant":
            messages.append(
                {"role": "assistant", "content": reply["text"]}
            )
            valid_sequences.append(messages[:])
            for child in reply["replies"]:
                dfs(child, messages, valid_sequences)
            messages.pop()
        elif reply["role"] == "prompter":
            messages.append(
                {"role": "user", "content": reply["text"]}
            )
            for child in reply["replies"]:
                dfs(child, messages, valid_sequences)
            messages.pop()
        else:
            raise ValueError(f"Unknown role: {reply['role']}")

    with open(output_path, "w") as fout:
        example_cnt = 0
        for _, conversation in enumerate(conversations):
            valid_sequences = []
            dfs(conversation["prompt"], [], valid_sequences)
            for sequence in valid_sequences:
                fout.write(json.dumps({
                    "dataset": "oasst1",
                    "id": f"oasst1_{example_cnt}",
                    "messages": sequence
                }) + "\n")
                example_cnt += 1

        

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--raw_data_dir", type=str, default="data/downloads")
    arg_parser.add_argument("--output_dir", type=str, default="data/processed")
    arg_parser.add_argument("--seed", type=int, default=42)
    args = arg_parser.parse_args()
    random.seed(args.seed)

    # get the subfolder names in raw_data_dir
    subfolders = [f for f in os.listdir(args.raw_data_dir) if os.path.isdir(os.path.join(args.raw_data_dir, f))]

    # all supported datasets    
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])

    # check if the subfolder names are supported datasets
    valid_subfolders = []
    for subfolder in subfolders:
        if subfolder not in supported_datasets:
            print(f"Warning: {subfolder} in the raw data folder is not a supported dataset. We will skip it.")
        else:
            valid_subfolders.append(subfolder)
    
    # prepare data for each dataset
    statistics = {}
    for subfolder in valid_subfolders:
        print(f"Processing {subfolder} data...")
        globals()[f"convert_{subfolder}_data"](os.path.join(args.raw_data_dir, subfolder), os.path.join(args.output_dir, subfolder))