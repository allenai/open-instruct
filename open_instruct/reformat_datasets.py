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
import pandas as pd
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
    if num_zero_shot_examples > 0:
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
            

def convert_flan_v2_data(data_dir, output_dir, data_file="tulu_v1_resampled_flan_100k.jsonl"):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
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


def convert_dolly_data(data_dir, output_dir, number_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "databricks-dolly-15k.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if number_examples:
        examples = random.sample(examples, k=number_examples)
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


def convert_self_instruct_data(data_dir, output_dir, number_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "all_instances_82K.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if number_examples:
        examples = random.sample(examples, k=number_examples)
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


def convert_unnatural_instructions_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "core_data.jsonl"), "r") as fin:
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
                    eos_token=None,
                )
                examples.append(encoded_example)
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    with open((os.path.join(output_dir, "unnatural_instructions_data.jsonl")), "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": "unnatural_instructions",
                "id": f"unnatural_instructions_{idx}",
                "messages": [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["completion"]},
                ]
            }) + "\n")


def convert_stanford_alpaca_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "alpaca_data.json"), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
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


def convert_code_alpaca_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "code_alpaca_20k.json"), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
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


def convert_gpt4_alpaca_data(data_dir, output_dir, load_en=True, load_zh=False, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    if load_en:
        with open(os.path.join(data_dir, "alpaca_gpt4_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if load_zh:
        with open(os.path.join(data_dir, "alpaca_gpt4_data_zh.json"), "r") as fin:
            examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
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


def convert_sharegpt_data(data_dir, output_dir, data_file="sharegpt_html_cleaned_and_split_2048.json", num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, data_file), "r") as fin:
        examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)

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
        if invalid_cnt > 0:
            print(f"# of invalid examples in sharegpt data: {invalid_cnt}")


def convert_baize_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    for source in ["alpaca", "medical", "quora", "stackoverflow"]:
        with open(os.path.join(data_dir, f"{source}_chat_data.json"), "r") as fin:
            examples.extend(json.load(fin))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
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


def convert_oasst1_data(data_dir, output_dir, top_k_reply=None):
    '''
    For OASST1, because it's in a tree structure, where every user input might get multiple replies, 
    we have to save every path from the root node to the assistant reply (including both leaf node and intemediate node).
    This results in some of the messages being duplicated among different paths (instances).
    You can set top_k_reply to control how many replies to consider when traversing the tree, which will consider the replies with 
    the highest human-reviewed quality scores.
    '''
    os.makedirs(output_dir, exist_ok=True)
    conversations = []
    with open(os.path.join(data_dir, "2023-04-12_oasst_ready.trees.jsonl"), "r") as fin:
        for line in fin:
            conversations.append(json.loads(line))

    output_path = os.path.join(output_dir, "oasst1_data.jsonl")

    # tranvers the conversation tree, and collect all valid sequences
    def dfs(reply, messages, valid_sequences):
        if reply["deleted"]:
            return
        if reply["role"] == "assistant":
            messages.append(
                {"role": "assistant", "content": reply["text"]}
            )
            if not reply["replies"]:  # leaf node
                valid_sequences.append(messages[:])
            else:
                child_replies = [child for child in reply["replies"] if not child["deleted"]]
                for child in child_replies:
                    if not "quality" in child["labels"]:
                        child["labels"]["quality"] = {
                            "value": 0.0,
                            "count": 0,
                        }
                child_replies = child_replies if top_k_reply is None else sorted(child_replies, key=lambda x: x["labels"]["quality"]["value"], reverse=True)[:top_k_reply]
                for child in child_replies:
                    dfs(child, messages, valid_sequences)
            messages.pop()
        elif reply["role"] == "prompter":
            messages.append(
                {"role": "user", "content": reply["text"]}
            )
            child_replies = [child for child in reply["replies"] if not child["deleted"]]
            for child in child_replies:
                if not "quality" in child["labels"]:
                    child["labels"]["quality"] = {
                        "value": 0.0,
                        "count": 0,
                    }
            child_replies = child_replies if top_k_reply is None else sorted(child_replies, key=lambda x: x["labels"]["quality"]["value"], reverse=True)[:top_k_reply]
            for child in child_replies:
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


def convert_lima_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "lima_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            messages = []
            if not len(example["conversations"]) % 2 == 0:
                print(f"Waring: example {idx} in LIMA has odd number of messages. Cutting off the last message.")
                example["conversations"] = example["conversations"][:-1]
            
            for i in range(0, len(example["conversations"]), 2):
                messages.append({
                    "role": "user",
                    "content": example["conversations"][i]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["conversations"][i+1]
                })
            fout.write(json.dumps({
                "dataset": "lima",
                "id": f"lima_{idx}",
                "messages": messages,
            }) + "\n")


def convert_wizardlm_data(data_dir, output_dir, num_examples=30000):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    # check if the original json file exists
    if os.path.exists(os.path.join(data_dir, "WizardLM_evol_instruct_V2_143k.json")):
        # proceed as normally
        with open(os.path.join(data_dir, "WizardLM_evol_instruct_V2_143k.json"), "r") as fin:
            examples = json.load(fin)
    else: # try other data train-00000-of-00001-004cd1ba9dc05e6c.parquet
        df = pd.read_parquet(os.path.join(data_dir, "train-00000-of-00001-004cd1ba9dc05e6c.parquet"))
        examples = df.to_dict(orient="records")
    if num_examples:
        examples = random.sample(examples, k=num_examples)

    output_path = os.path.join(output_dir, "wizardlm_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            messages = []
            assert len(example["conversations"]) % 2 == 0
            for i in range(0, len(example["conversations"]), 2):
                assert example["conversations"][i]["from"] == "human"
                assert example["conversations"][i+1]["from"] == "gpt"
                messages.append({
                    "role": "user",
                    "content": example["conversations"][i]["value"]
                })
                messages.append({
                    "role": "assistant",
                    "content": example["conversations"][i+1]["value"]
                })
            fout.write(json.dumps({
                "dataset": "wizardlm",
                "id": f"wizardlm_{example['idx']}",
                "messages": messages,
            }) + "\n")


def convert_open_orca_data(data_dir, output_dir, num_gpt4_examples=30000, num_gpt35_examples=0):
    os.makedirs(output_dir, exist_ok=True)
    examples = []

    df = pd.read_parquet(os.path.join(data_dir, "1M-GPT4-Augmented.parquet"))    
    gpt4_examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(gpt4_examples)
    examples.extend(gpt4_examples[:num_gpt4_examples])

    df = pd.read_parquet(os.path.join(data_dir, "3_5M-GPT3_5-Augmented.parquet"))
    gpt35_examples = [row.to_dict() for _, row in df.iterrows()]
    random.shuffle(gpt35_examples)
    examples.extend(gpt35_examples[:num_gpt35_examples])

    output_path = os.path.join(output_dir, "open_orca_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            messages = [
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["response"]}
            ]
            fout.write(json.dumps({
                "dataset": "open_orca",
                "id": f"open_orca_{example['id']}",
                "messages": messages,
            }) + "\n")


def convert_hard_coded_data(data_dir, output_dir, repeat=1):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_excel(os.path.join(data_dir, "hard_coded_examples.xlsx"), header=0)
    output_path = os.path.join(output_dir, "hard_coded_data.jsonl")
    with open(output_path, "w") as fout:
        for _ in range(repeat):
            for idx, row in data.iterrows():
                fout.write(json.dumps({
                    "dataset": "hard_coded",
                    "id": f"hard_coded_{idx}",
                    "messages": [
                        {"role": "user", "content": row["Prompt"]},
                        {"role": "assistant", "content": row["Output"]}
                    ]
                }) + "\n")


def convert_science_data(data_dir, output_dir, num_examples=None):
    os.makedirs(output_dir, exist_ok=True)
    examples = []
    with open(os.path.join(data_dir, "science_train.jsonl"), "r") as fin:
        for line in fin:
            examples.append(json.loads(line))
    if num_examples:
        examples = random.sample(examples, k=num_examples)
    output_path = os.path.join(output_dir, "science_data.jsonl")
    with open(output_path, "w") as fout:
        for idx, example in enumerate(examples):
            fout.write(json.dumps({
                "dataset": f"science.{example['dataset']}",
                "id": f"science_{idx}",
                "messages": [
                    {"role": "user", "content": example["input"]},
                    {"role": "assistant", "content": example["output"]}
                ],
            }) + "\n")



def should_be_filtered(example):
    # we filter out conversations that contain some specific strings
    filter_strings = [
        "OpenAI",
        "Open AI",
        "ChatGPT",
        "Chat GPT",
        "GPT-3",
        "GPT3",
        "GPT 3",
        "GPT-4",
        "GPT4",
        "GPT 4",
        "GPT-3.5",
        "GPT3.5",
        "GPT 3.5",
        "BingChat",
        "Bing Chat",
        "BARD",
        "Palm",
        "Anthropic",
        "Claude",
        "LAION",
        "Open Assistant",
        "OpenAssistant", 
    ]
    for message in example["messages"]:
        if any([filter_string.lower() in message["content"].lower() for filter_string in filter_strings]):
            return True
    return False
        

if __name__ == "__main__":
    # all supported datasets    
    supported_datasets = []
    all_funcs = [func_name for func_name in globals() if callable(globals()[func_name])]
    for func_name in all_funcs:
        if re.match(r"convert_.+_data", func_name):
            supported_datasets.append(func_name[8:-5])

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--raw_data_dir", 
        type=str, 
        default="data/downloads"
    )
    arg_parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/processed"
    )
    arg_parser.add_argument(
        "--dataset", 
        type=str, 
        nargs="+",
        choices=supported_datasets+["tulu_v1", "tulu_v2"],
        default=supported_datasets+["tulu_v1", "tulu_v2"]
    )
    arg_parser.add_argument(
        "--seed", 
        type=int, 
        default=42
    )
    args = arg_parser.parse_args()
    random.seed(args.seed)

    # get the subfolder names in raw_data_dir
    subfolders = [f for f in os.listdir(args.raw_data_dir) if os.path.isdir(os.path.join(args.raw_data_dir, f))]

    for dataset in args.dataset:
        if dataset == "tulu_v1":
            print(f"Processing tulu_v1 subsets...")
            convert_flan_v2_data(
                data_dir=os.path.join(args.raw_data_dir, "flan_v2"), 
                output_dir=os.path.join(args.output_dir, "tulu_v1", "flan_v2_subset"),
                data_file="tulu_v1_resampled_flan_100k.jsonl",
            )
            convert_cot_data(
                data_dir=os.path.join(args.raw_data_dir, "cot"), 
                output_dir=os.path.join(args.output_dir, "tulu_v1", "cot_subset"),
                num_few_shot_examples=50000,
                num_zero_shot_examples=50000
            )
            convert_oasst1_data(
                data_dir=os.path.join(args.raw_data_dir, "oasst1"), 
                output_dir=os.path.join(args.output_dir, "tulu_v1", "oasst1_subset"), 
                top_k_reply=None
            )
            convert_dolly_data(
                data_dir=os.path.join(args.raw_data_dir, "dolly"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "dolly_subset"),
            )
            convert_gpt4_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "gpt4_alpaca"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "gpt4_alpaca_subset"),
                load_en=True,
                load_zh=False,
                num_examples=None
            )
            convert_code_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "code_alpaca"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "code_alpaca_subset"),
                num_examples=None
            )
            convert_sharegpt_data(
                data_dir=os.path.join(args.raw_data_dir, "sharegpt"),
                output_dir=os.path.join(args.output_dir, "tulu_v1", "sharegpt_subset"),
                data_file="sharegpt_html_cleaned_and_split_2048.json",
                num_examples=None
            )
            # merge all the subsets
            print("Merging all the subsets to create tulu v1...")
            all_subsets = [f for f in os.listdir(os.path.join(args.output_dir, "tulu_v1")) if f.endswith("_subset")]
            with open(os.path.join(args.output_dir, "tulu_v1", "tulu_v1_data.jsonl"), "w") as fout:
                for subset in all_subsets:
                    dataset_name = subset[:-len("_subset")]
                    with open(os.path.join(args.output_dir, "tulu_v1", subset, f"{dataset_name}_data.jsonl"), "r") as fin:
                        for line in fin:
                            fout.write(line)
        elif dataset == "tulu_v2":
            print(f"Processing tulu_v2 subsets...")
            convert_flan_v2_data(
                data_dir=os.path.join(args.raw_data_dir, "flan_v2"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "flan_v2_subset"),
                data_file="tulu_v2_resampled_flan_50k.jsonl",
            )
            convert_cot_data(
                data_dir=os.path.join(args.raw_data_dir, "cot"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "cot_subset"),
                num_few_shot_examples=25000,
                num_zero_shot_examples=25000
            )
            convert_oasst1_data(
                data_dir=os.path.join(args.raw_data_dir, "oasst1"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "oasst1_subset"), 
                top_k_reply=1
            )
            convert_lima_data(
                data_dir=os.path.join(args.raw_data_dir, "lima"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "lima_subset"), 
                num_examples=None
            )
            convert_gpt4_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "gpt4_alpaca"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "gpt4_alpaca_subset"), 
                load_en=True, 
                load_zh=False, 
                num_examples=20000
            )
            convert_code_alpaca_data(
                data_dir=os.path.join(args.raw_data_dir, "code_alpaca"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "code_alpaca_subset"), 
                num_examples=None
            )
            convert_sharegpt_data(
                data_dir=os.path.join(args.raw_data_dir, "sharegpt"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "sharegpt_subset"),
                data_file="sharegpt_html_cleaned_and_split_4096.json",
                num_examples=None
            )
            convert_wizardlm_data(
                data_dir=os.path.join(args.raw_data_dir, "wizardlm"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "wizardlm_subset"), 
                num_examples=30000
            )
            convert_open_orca_data(
                data_dir=os.path.join(args.raw_data_dir, "open_orca"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "open_orca_subset"), 
                num_gpt4_examples=30000, 
                num_gpt35_examples=0
            )
            convert_science_data(
                data_dir=os.path.join(args.raw_data_dir, "science"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "science_subset"),
                num_examples=None
            )
            convert_hard_coded_data(
                data_dir=os.path.join(args.raw_data_dir, "hard_coded"), 
                output_dir=os.path.join(args.output_dir, "tulu_v2", "hard_coded_subset"),
                repeat=10,
            )
            # merge all the subsets
            print("Merging all the subsets to create tulu v2...")
            all_subsets = [f for f in os.listdir(os.path.join(args.output_dir, "tulu_v2")) if f.endswith("_subset")]
            with open(os.path.join(args.output_dir, "tulu_v2", "tulu_v2_data.jsonl"), "w") as fout, \
                open(os.path.join(args.output_dir, "tulu_v2", "tulu_v2_filtered_data.jsonl"), "w") as fout_filtered:
                for subset in all_subsets:
                    dataset_name = subset[:-len("_subset")]
                    with open(os.path.join(args.output_dir, "tulu_v2", subset, f"{dataset_name}_data.jsonl"), "r") as fin:
                        for line in fin:
                            example = json.loads(line)
                            if subset not in ["hard_coded_subset"] and should_be_filtered(example):
                                fout_filtered.write(line)
                            else:
                                fout.write(line)
        else:
            print(f"Processing {dataset} data with default configurations...")
            globals()[f"convert_{dataset}_data"](os.path.join(args.raw_data_dir, dataset), os.path.join(args.output_dir, dataset))