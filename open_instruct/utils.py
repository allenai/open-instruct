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
# isort: off
import os

os.environ["NCCL_CUMEM_ENABLE"] = "0"  # NOQA
try:
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    from deepspeed.runtime.zero.offload_config import OffloadDeviceEnum

    # @vwxyzjn: when importing on CPU-only machines, we get the following error:
    # RuntimeError: 0 active drivers ([]). There should only be one.
    # so we need to catch the exception and do nothing
    # https://github.com/deepspeedai/DeepSpeed/issues/7028
except Exception:
    pass
# isort: on
import dataclasses
import functools
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import time
from ctypes import CDLL, POINTER, Structure, c_char_p, c_int, c_ulong, c_void_p
from dataclasses import dataclass
from typing import Any, List, NewType, Optional, Tuple, Union

import numpy as np
import ray
import requests
import torch
from accelerate.logging import get_logger
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from dateutil import parser
from huggingface_hub import HfApi
from rich.pretty import pprint
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

logger = get_logger(__name__)

DataClassType = NewType("DataClassType", Any)

"""
Notes:
Inspired by Alignment Handbook Parser and Dataset Mixer
https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/configs.py
https://github.com/huggingface/alignment-handbook/blob/main/src/alignment/data.py

Migrated Args from
https://github.com/allenai/open-instruct/blob/98ccfb460ae4fb98140783b6cf54241926160a06/open_instruct/finetune_trainer.py

Commented out Args not currently used
"""


# ----------------------------------------------------------------------------
# Dataset utilities
def is_openai_format(messages: Any) -> bool:
    """
    Check if the input messages are in OpenAI format.
    Args:
        messages (`Any`):
            Messages to check.
    Returns:
        `bool`: Whether the messages are in OpenAI format.
    """
    if isinstance(messages, list) and all(isinstance(message, dict) for message in messages):
        return all("role" in message and "content" in message for message in messages)
    return False


# functions for handling different formats of messages
def convert_alpaca_gpt4_to_messages(example):
    """
    Convert an instruction in inst-output to a list of messages.
    e.g. vicgalle/alpaca-gpt4"""
    messages = [
        {
            "role": "user",
            "content": (
                "Below is an instruction that describes a task, paired with an input that provides "
                "further context. Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                "### Response:"
            ),
        },
        {"role": "assistant", "content": example["output"]},
    ]
    example["messages"] = messages
    return example


def convert_codefeedback_single_turn_to_messages(example):
    """
    Convert a query-answer pair to a list of messages.
    e.g. m-a-p/CodeFeedback-Filtered-Instruction"""
    messages = [{"role": "user", "content": example["query"]}, {"role": "assistant", "content": example["answer"]}]
    example["messages"] = messages
    return example


def convert_metamath_qa_to_messages(example):
    """
    Convert a query-response pair to a list of messages.
    e.g. meta-math/MetaMathQA"""
    messages = [{"role": "user", "content": example["query"]}, {"role": "assistant", "content": example["response"]}]
    example["messages"] = messages
    return example


def convert_code_alpaca_to_messages(example):
    """
    Convert a prompt-completion pair to a list of messages.
    e.g. HuggingFaceH4/CodeAlpaca_20K"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    example["messages"] = messages
    return example


def convert_open_orca_to_messages(example):
    """
    Convert a question-response pair to a list of messages.
    e.g. Open-Orca/OpenOrca"""
    messages = [
        {"role": "system", "content": example["system_prompt"]},
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example


def conversations_to_messages(example):
    """
    Convert from conversations format to messages.

    E.g. change "from": "user" to "role": "user"
        and "value" to "content"
        and "gpt" to "assistant"

    WizardLMTeam/WizardLM_evol_instruct_V2_196k
    """
    name_mapping = {
        "gpt": "assistant",
        "Assistant": "assistant",
        "assistant": "assistant",
        "user": "user",
        "User": "user",
        "human": "user",
    }
    messages = [{"role": name_mapping[conv["from"]], "content": conv["value"]} for conv in example["conversations"]]
    example["messages"] = messages
    return example


def convert_rejection_samples_to_messages(example):
    """
    Convert a rejection sampling dataset to messages.
    """
    example["messages"] = example["chosen"]
    return example


def get_datasets(
    dataset_mixer: Union[dict, list],
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    save_data_dir: Optional[str] = None,
    need_columns: Optional[List[str]] = None,
    keep_ids: bool = False,
    add_source_col: bool = False,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`list` or `dict`):
            Dictionary or list containing the dataset names and their training proportions.
            By default, all test proportions are 1. Lists are formatted as
            `key1 value1 key2 value2 ...` If a list is passed in, it will be converted to a dictionary.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in
            all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `True`):
            Whether to shuffle the training and testing/validation data.
        save_data_dir (Optional[str], *optional*, defaults to `None`):
            Optional directory to save training/test mixes on.
        need_columns (Optional[List[str]], *optional*, defaults to `None`):
            Column names that are required to be in the dataset.
            Quick debugging when mixing heterogeneous datasets.
        keep_ids (`bool`, *optional*, defaults to `False`):
            Whether to keep ids for training that are added during mixing.
            Used primarily in mix_data.py for saving, or the saved dataset has IDs already.
        add_source_col (`bool`, *optional*, defaults to `False`):
            Whether to add a column to the dataset that indicates the source of the data explicitly.
    """
    if isinstance(dataset_mixer, list):
        assert len(dataset_mixer) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer}"
        mixer_dict = {}
        i = 0
        while i < len(dataset_mixer) - 1:
            assert isinstance(dataset_mixer[i], str), f"Invalid type in data mixer: {dataset_mixer}"
            if "." in dataset_mixer[i + 1]:
                value = float(dataset_mixer[i + 1])
            else:
                value = int(dataset_mixer[i + 1])
            mixer_dict[dataset_mixer[i]] = value
            i += 2
        dataset_mixer = mixer_dict

    splits = ["train", "test"] if splits is None else splits
    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    # print save location
    if save_data_dir:
        print(f"Saving mixed dataset to {save_data_dir}")

    raw_datasets = DatasetDict()
    raw_train_datasets = []
    raw_val_datasets = []
    frac_or_sample_list = []
    for (ds, frac_or_samples), ds_config in zip(dataset_mixer.items(), configs):
        frac_or_sample_list.append(frac_or_samples)
        for split in splits:
            # if dataset ends with .json or .jsonl, load from file
            if ds.endswith(".json") or ds.endswith(".jsonl"):
                dataset = load_dataset("json", data_files=ds, split=split)
            else:
                try:
                    # Try first if dataset on a Hub repo
                    dataset = load_dataset(ds, ds_config, split=split)
                except DatasetGenerationError:
                    # If not, check local dataset
                    dataset = load_from_disk(os.path.join(ds, split))

            # shuffle dataset if set
            if shuffle:
                dataset = dataset.shuffle(seed=42)

            # assert that needed columns are present
            if need_columns:
                if not all(col in dataset.column_names for col in need_columns):
                    raise ValueError(f"Needed column {need_columns} not found in dataset {dataset.column_names}.")

            # handle per-case conversions
            # if "instruction" and "output" columns are present and "messages" is not, convert to messages
            if (
                "instruction" in dataset.column_names
                and "output" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_alpaca_gpt4_to_messages, num_proc=10)
            elif (
                "prompt" in dataset.column_names
                and "completion" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_code_alpaca_to_messages, num_proc=10)
            elif "conversations" in dataset.column_names and "messages" not in dataset.column_names:
                dataset = dataset.map(conversations_to_messages, num_proc=10)
            elif (
                "question" in dataset.column_names
                and "response" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_open_orca_to_messages, num_proc=10)
            elif (
                "query" in dataset.column_names
                and "answer" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_codefeedback_single_turn_to_messages, num_proc=10)
            elif (
                "query" in dataset.column_names
                and "response" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_metamath_qa_to_messages, num_proc=10)
            elif (
                "chosen" in dataset.column_names
                and "rejected" in dataset.column_names
                and "reference_completion" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(convert_rejection_samples_to_messages, num_proc=10)

            # if id not in dataset, create it as ds-{index}
            if "id" not in dataset.column_names:
                id_col = [f"{ds}_{i}" for i in range(len(dataset))]
                dataset = dataset.add_column("id", id_col)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in (columns_to_keep + ["id"])]
            )

            # if add_source_col, add that column
            if add_source_col:
                source_col = [ds] * len(dataset)
                dataset = dataset.add_column("source", source_col)

            # for cols in columns_to_keep, if one is not present, add "None" to the column
            for col in columns_to_keep:
                if col not in dataset.column_names:
                    dataset = dataset.add_column(col, [None] * len(dataset))

            # add tag to the dataset corresponding to where it was sourced from, for
            if "train" in split:
                raw_train_datasets.append(dataset)
            elif "test" in split:
                raw_val_datasets.append(dataset)
            else:
                raise ValueError(f"Split type {split} not recognized as one of test or train.")

    if len(raw_val_datasets) == 0 and len(raw_train_datasets) == 0:
        raise ValueError("No datasets loaded.")
    elif len(raw_train_datasets) == 0:
        # target features are the features of the first dataset post load
        target_features = raw_val_datasets[0].features
    else:
        # target features are the features of the first dataset post load
        target_features = raw_train_datasets[0].features

    if any(frac_or_samples < 0 for frac_or_samples in frac_or_sample_list):
        raise ValueError("Dataset fractions / lengths cannot be negative.")

    # if any > 1, use count
    if any(frac_or_samples > 1 for frac_or_samples in frac_or_sample_list):
        is_count = True
        # assert that all are integers
        if not all(isinstance(frac_or_samples, int) for frac_or_samples in frac_or_sample_list):
            raise NotImplementedError("Cannot mix fractions and counts, yet.")
    else:
        is_count = False

    if len(raw_train_datasets) > 0:
        train_subsets = []
        # Manage proportions
        for dataset, frac_or_samples in zip(raw_train_datasets, frac_or_sample_list):
            # cast features (TODO, add more feature regularization)
            dataset = dataset.cast(target_features)
            # TODO selection can be randomized.
            if is_count:
                train_subset = dataset.select(range(frac_or_samples))
            else:
                train_subset = dataset.select(range(int(frac_or_samples * len(dataset))))
            train_subsets.append(train_subset)

        raw_datasets["train"] = concatenate_datasets(train_subsets)

    # No subsampling for test datasets to enable fair comparison across models
    if len(raw_val_datasets) > 0:
        for dataset in raw_val_datasets:
            # cast features (TODO, add more feature regularization)
            dataset = dataset.cast(target_features)

        raw_datasets["test"] = concatenate_datasets(raw_val_datasets)

    if len(raw_datasets) == 0:
        raise ValueError(
            f"Dataset {dataset_mixer} not recognized with splits {splits}."
            "Check the dataset has been correctly formatted."
        )

    # optional save
    if save_data_dir:
        for split in raw_datasets:
            raw_datasets[split].to_json(save_data_dir + f"mixed_ds_{split}.json")

    if not keep_ids:
        # remove id column
        if len(raw_train_datasets) > 0:
            if "id" in raw_datasets["train"].column_names:
                raw_datasets["train"] = raw_datasets["train"].remove_columns("id")
        if len(raw_val_datasets) > 0:
            if "id" in raw_datasets["test"].column_names:
                raw_datasets["test"] = raw_datasets["test"].remove_columns("id")

    return raw_datasets


def combine_dataset(
    dataset_mixer: Union[dict, list],
    splits: List[str],
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = False,
    save_data_dir: Optional[str] = None,
    keep_ids: bool = False,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions.
        splits (Optional[List[str]], *optional*, defaults to `None`):
            Dataset splits to load and mix. Assumes the splits exist in
            all datasets and have a `train_` or `test_` prefix.
        configs (Optional[List[str]], *optional*, defaults to `None`):
            List of dataset config names. If given must be the same length as 'dataset_mixer' keys.
        columns_to_keep (Optional[List[str]], *optional*, defaults to `None`):
            Column names to keep in the dataset. Useful in the datamixer to avoid schema conflicts,
            and for cpt this should be (at least) the text column.
        shuffle (`bool`, *optional*, defaults to `False`):
            Whether to shuffle the training and testing/validation data.
        save_data_dir (Optional[str], *optional*, defaults to `None`):
            Optional directory to save training/test mixes on.
        keep_ids (`bool`, *optional*, defaults to `False`):
            Whether to keep ids for training that are added during mixing.
            Used primarily in mix_data.py for saving, or the saved dataset has IDs already.
    """
    assert len(splits) == len(dataset_mixer), "Number of splits must match the number of datasets."
    if isinstance(dataset_mixer, list):
        assert len(dataset_mixer) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer}"
        mixer_dict = {}
        i = 0
        while i < len(dataset_mixer) - 1:
            assert isinstance(dataset_mixer[i], str), f"Invalid type in data mixer: {dataset_mixer}"
            if "." in dataset_mixer[i + 1]:
                value = float(dataset_mixer[i + 1])
            else:
                value = int(dataset_mixer[i + 1])
            mixer_dict[dataset_mixer[i]] = value
            i += 2
        dataset_mixer = mixer_dict

    if any(frac_or_samples < 0 for frac_or_samples in dataset_mixer.values()):
        raise ValueError("Dataset fractions / lengths cannot be negative.")

    configs = [None] * len(dataset_mixer) if not configs else configs
    columns_to_keep = [] if columns_to_keep is None else columns_to_keep

    if configs is not None and len(configs) != len(dataset_mixer):
        raise ValueError("The number of given dataset config names must be the same as the given number of datasets.")

    # print save location
    if save_data_dir:
        print(f"Saving mixed dataset to {save_data_dir}")

    datasets = []
    for (ds, frac_or_samples), ds_config, split in zip(dataset_mixer.items(), configs, splits):
        # if dataset ends with .json or .jsonl, load from file
        if ds.endswith(".json") or ds.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=ds, split=split)
        else:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split)
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

        # shuffle dataset if set
        if shuffle:
            dataset = dataset.shuffle(seed=42)

        # select a fraction of the dataset
        if frac_or_samples > 1.0:
            samples = int(frac_or_samples)
        else:
            samples = int(frac_or_samples * len(dataset))
        dataset = dataset.select(range(samples))

        # if id not in dataset, create it as ds-{index}
        if "id" not in dataset.column_names:
            id_col = [f"{ds}_{i}_{split}" for i in range(len(dataset))]
            dataset = dataset.add_column("id", id_col)

        # Remove redundant columns to avoid schema conflicts on load
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in (columns_to_keep + ["id"])]
        )
        datasets.append(dataset)

    datasets = concatenate_datasets(datasets)

    # optional save
    if save_data_dir:
        datasets.to_json(save_data_dir + "mixed_ds.json")

    if not keep_ids:
        # remove id column
        if "id" in datasets.column_names:
            datasets = datasets.remove_columns("id")

    return datasets


# ----------------------------------------------------------------------------
# Arguments utilities
class ArgumentParserPlus(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # noqa adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> Union[DataClassType, Tuple[DataClassType]]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


# ----------------------------------------------------------------------------
# Experiment tracking utilities
def get_git_tag() -> str:
    """Try to get the latest Git tag (e.g., `no-tag-404-g98dc659` or `v1.0.0-4-g98dc659`)"""
    git_tag = ""
    try:
        git_tag = (
            subprocess.check_output(["git", "describe", "--tags"], stderr=subprocess.DEVNULL).decode("ascii").strip()
        )
    except subprocess.CalledProcessError as e:
        logging.debug(f"Failed to get Git tag: {e}")

    # If no Git tag found, create a custom tag based on commit count and hash
    if len(git_tag) == 0:
        try:
            count = int(
                subprocess.check_output(["git", "rev-list", "--count", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("ascii")
                .strip()
            )
            hash = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL)
                .decode("ascii")
                .strip()
            )
            git_tag = f"no-tag-{count}-g{hash}"
        except subprocess.CalledProcessError as e:
            logging.debug(f"Failed to get commit count and hash: {e}")

    return git_tag


def get_pr_tag() -> str:
    """Try to find associated pull request on GitHub (e.g., `pr-123`)"""
    pr_tag = ""
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        # try finding the pull request number on github
        prs = requests.get(f"https://api.github.com/search/issues?q=repo:allenai/open-instruct+is:pr+{git_commit}")
        if prs.status_code == 200:
            prs = prs.json()
            if len(prs["items"]) > 0:
                pr = prs["items"][0]
                pr_number = pr["number"]
                pr_tag = f"pr-{pr_number}"
    except Exception as e:
        logging.debug(f"Failed to get PR number: {e}")

    return pr_tag


def get_wandb_tags() -> List[str]:
    """Get tags for Weights & Biases (e.g., `no-tag-404-g98dc659,pr-123`)"""
    existing_wandb_tags = os.environ.get("WANDB_TAGS", "")
    git_tag = get_git_tag()
    pr_tag = get_pr_tag()
    non_empty_tags = [tag for tag in [existing_wandb_tags, git_tag, pr_tag] if len(tag) > 0]
    return non_empty_tags


# ----------------------------------------------------------------------------
# Check pointing utilities
def get_last_checkpoint(folder: str, incomplete: bool = False) -> Optional[str]:
    content = os.listdir(folder)
    checkpoint_steps = [path for path in content if path.startswith("step_")]
    checkpoint_epochs = [path for path in content if path.startswith("epoch_")]
    if len(checkpoint_steps) > 0 and len(checkpoint_epochs) > 0:
        logger.info("Mixed step and epoch checkpoints found. Using step checkpoints.")
        checkpoints = checkpoint_steps
    elif len(checkpoint_steps) == 0:
        checkpoints = checkpoint_epochs
    else:
        checkpoints = checkpoint_steps
    if not incomplete:
        checkpoints = [path for path in checkpoints if os.path.exists(os.path.join(folder, path, "COMPLETED"))]
    if len(checkpoints) == 0:
        return
    return os.path.join(folder, max(checkpoints, key=lambda x: int(x.split("_")[-1])))


def get_last_checkpoint_path(args, incomplete: bool = False) -> str:
    # if output already exists and user does not allow overwriting, resume from there.
    # otherwise, resume if the user specifies a checkpoint.
    # else, start from scratch.
    # if incomplete is true, include folders without "COMPLETE" in the folder.
    last_checkpoint_path = None
    if args.output_dir and os.path.isdir(args.output_dir):
        last_checkpoint_path = get_last_checkpoint(args.output_dir, incomplete=incomplete)
        if last_checkpoint_path is None:
            logger.warning("Output directory exists but no checkpoint found. Starting from scratch.")
    elif args.resume_from_checkpoint:
        last_checkpoint_path = args.resume_from_checkpoint
    return last_checkpoint_path


def is_checkpoint_folder(dir: str, folder: str) -> bool:
    return (folder.startswith("step_") or folder.startswith("epoch_")) and os.path.isdir(os.path.join(dir, folder))


def clean_last_n_checkpoints(output_dir: str, keep_last_n_checkpoints: int) -> None:
    # remove the last checkpoint to save space
    folders = [f for f in os.listdir(output_dir) if is_checkpoint_folder(output_dir, f)]
    # find the checkpoint with the largest step
    checkpoints = sorted(folders, key=lambda x: int(x.split("_")[-1]))
    if keep_last_n_checkpoints >= 0 and len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[: len(checkpoints) - keep_last_n_checkpoints]:
            logger.info(f"Removing checkpoint {checkpoint}")
            shutil.rmtree(os.path.join(output_dir, checkpoint))
    logger.info("Remaining files:" + str(os.listdir(output_dir)))


def clean_last_n_checkpoints_deepspeed(output_dir: str, keep_last_n_checkpoints: int) -> None:
    # Identify checkpoint files that follow the pattern global_step{number}
    all_files = os.listdir(output_dir)
    checkpoint_files = []
    for file in all_files:
        if file.startswith("global_step") and file[len("global_step") :].isdigit():
            checkpoint_files.append(file)

    # Sort checkpoints by step number
    checkpoints = sorted(checkpoint_files, key=lambda x: int(x[len("global_step") :]), reverse=True)

    # Keep the N most recent checkpoints and remove the rest
    if keep_last_n_checkpoints >= 0 and len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[keep_last_n_checkpoints:]:
            print(f"Removing checkpoint {checkpoint}")
            checkpoint_path = os.path.join(output_dir, checkpoint)
            if os.path.isdir(checkpoint_path):
                shutil.rmtree(checkpoint_path)
            elif os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)

    # Keep special files like zero_to_fp32.py and latest
    print("Remaining files:" + str(os.listdir(output_dir)))


def calibrate_checkpoint_state_dir(checkpoint_state_dir: str) -> None:
    """
    Find the latest valid checkpoint directory and update the 'latest' file.

    Edge case:
    it's possible sometimes the checkpoint save / upload (1) completely or (2) partially failed (i.e., having incomplete files),
    so we should fall back to a checkpoint that actually exists -- we should pick the latest folder which has the most files.
    The folders look like this:
    checkpoint_state_dir/global_step14
    checkpoint_state_dir/global_step15
    ...
    checkpoint_state_dir/global_step20
    we would then update the `checkpoint_state_dir/latest` file
    with the latest global_step number.
    """
    if not os.path.exists(checkpoint_state_dir):
        return

    # Get all checkpoint directories
    checkpoint_dirs = [
        d
        for d in os.listdir(checkpoint_state_dir)
        if d.startswith("global_step") and os.path.isdir(os.path.join(checkpoint_state_dir, d))
    ]

    if not checkpoint_dirs:
        return

    # Create a list of (dir_name, step_number, file_count) tuples
    checkpoint_info = []
    for dir_name in checkpoint_dirs:
        step_number = int(dir_name.replace("global_step", ""))
        dir_path = os.path.join(checkpoint_state_dir, dir_name)
        # Count files in the directory, not directories
        file_count = len(os.listdir(dir_path))
        checkpoint_info.append((dir_name, step_number, file_count))

    # Find the maximum file count
    max_file_count = max(info[2] for info in checkpoint_info)

    # Filter to only include checkpoints with the maximum file count
    valid_checkpoints = [info for info in checkpoint_info if info[2] >= max_file_count]
    invalid_checkpoints = [info for info in checkpoint_info if info[2] < max_file_count]

    # Remove invalid checkpoint directories
    for dir_name, _, _ in invalid_checkpoints:
        checkpoint_path = os.path.join(checkpoint_state_dir, dir_name)
        print(f"Removing incomplete checkpoint: {dir_name}")
        shutil.rmtree(checkpoint_path)

    # Sort by step number (descending)
    valid_checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Get the latest valid checkpoint
    latest_checkpoint, latest_step, file_count = valid_checkpoints[0]

    # Update the 'latest' file
    with open(os.path.join(checkpoint_state_dir, "latest"), "w") as f:
        f.write(f"global_step{latest_step}")

    print(
        f"Found latest checkpoint: {latest_checkpoint} with {file_count} files, "
        f"updated 'latest' file to global_step{latest_step}"
    )


# ----------------------------------------------------------------------------
# Ai2 user utilities
@dataclass
class BeakerRuntimeConfig:
    beaker_workload_id: str
    beaker_node_hostname: Optional[List[str]] = None
    beaker_experiment_url: Optional[List[str]] = None
    beaker_dataset_ids: Optional[List[str]] = None
    beaker_dataset_id_urls: Optional[List[str]] = None


def is_beaker_job() -> bool:
    return "BEAKER_JOB_ID" in os.environ


def get_beaker_experiment_info(experiment_id: str) -> Optional[dict]:
    get_experiment_command = f"beaker experiment get {experiment_id} --format json"
    process = subprocess.Popen(["bash", "-c", get_experiment_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Failed to get Beaker experiment: {stderr}")
        return None
    return json.loads(stdout)[0]


def beaker_experiment_succeeded(experiment_id: str) -> bool:
    experiment = get_beaker_experiment_info(experiment_id)
    if "replicas" in experiment["jobs"][0]["execution"]["spec"]:
        num_replicas = experiment["jobs"][0]["execution"]["spec"]["replicas"]
    else:
        num_replicas = 1
    if not experiment:
        return False
    pprint(experiment)
    finalizeds = [
        "finalized" in job["status"] and "exitCode" in job["status"] and job["status"]["exitCode"] == 0
        for job in experiment["jobs"]
    ]
    pprint(finalizeds)
    return sum(finalizeds) == num_replicas


@dataclass
class DatasetInfo:
    id: str
    committed: Any
    non_empty: bool


def get_beaker_dataset_ids(experiment_id: str, sort=False) -> Optional[List[str]]:
    """if sort is True, the non-empty latest dataset will be availble at the end of the list"""
    experiment = get_beaker_experiment_info(experiment_id)
    if not experiment:
        return None
    result_ids = [job["result"]["beaker"] for job in experiment["jobs"]]
    dataset_infos = []
    for result_id in result_ids:
        get_dataset_command = f"beaker dataset get {result_id} --format json"
        process = subprocess.Popen(["bash", "-c", get_dataset_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Failed to get Beaker dataset: {stderr}")
            return None
        datasets = json.loads(stdout)
        dataset_infos.extend(
            [
                DatasetInfo(
                    id=dataset["id"],
                    committed=dataset["committed"],
                    non_empty=(
                        False if dataset["storage"]["totalSize"] is None else dataset["storage"]["totalSize"] > 0
                    ),
                )
                for dataset in datasets
            ]
        )
    if sort:
        # sort based on empty, then commited
        dataset_infos.sort(key=lambda x: (x.non_empty, parser.parse(x.committed)))
    pprint(dataset_infos)
    return [dataset.id for dataset in dataset_infos]


@functools.lru_cache(maxsize=1)
def get_beaker_whoami() -> Optional[str]:
    get_beaker_whoami_command = "beaker account whoami --format json"
    process = subprocess.Popen(
        ["bash", "-c", get_beaker_whoami_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Failed to get Beaker account: {stderr}")
        return None
    accounts = json.loads(stdout)
    return accounts[0]["name"]


def maybe_get_beaker_config():
    beaker_dataset_ids = get_beaker_dataset_ids(os.environ["BEAKER_WORKLOAD_ID"])
    # fix condition on basic interactive jobs
    if beaker_dataset_ids is None:
        beaker_dataset_id_urls = []
    else:
        beaker_dataset_id_urls = [f"https://beaker.org/ds/{dataset_id}" for dataset_id in beaker_dataset_ids]
    return BeakerRuntimeConfig(
        beaker_workload_id=os.environ["BEAKER_WORKLOAD_ID"],
        beaker_node_hostname=os.environ["BEAKER_NODE_HOSTNAME"],
        beaker_experiment_url=f"https://beaker.org/ex/{os.environ['BEAKER_WORKLOAD_ID']}/",
        beaker_dataset_ids=get_beaker_dataset_ids(os.environ["BEAKER_WORKLOAD_ID"]),
        beaker_dataset_id_urls=beaker_dataset_id_urls,
    )


def live_subprocess_output(cmd: List[str]) -> str:
    output_lines = []
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    # Display output in real-time and collect it
    for line in iter(process.stdout.readline, ""):
        if line.strip():
            print(line.strip())
            output_lines.append(line.strip())
    process.wait()
    if process.returncode != 0:
        # Get the actual error message from the process
        process_error = process.stderr.read() if process.stderr else "No error message available"
        error_message = f"gsutil command failed with return code {process.returncode}: {process_error}"
        print(error_message)
        raise Exception(error_message)

    return "\n".join(output_lines)


def download_from_hf(model_name_or_path: str, revision: str) -> None:
    cmd = ["huggingface-cli", "download", model_name_or_path, "--revision", revision]
    print(f"Downloading from HF with command: {cmd}")
    output = live_subprocess_output(cmd)
    # for some reason, sometimes the output includes the line including some loading message.
    # so do some minor cleaning.
    if "\n" in output:
        output = output.split("\n")[-1].strip()
    return output


[
    "gsutil",
    "-o",
    "GSUtil:parallel_composite_upload_threshold=150M",
    "cp",
    "-r",
    "/root/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5",
    "gs://ai2-llm/post-training/deletable_cache_models/Qwen/Qwen3-8B/9c925d64d72725edaf899c6cb9c377fd0709d9c5",
]


def download_from_gs_bucket(src_path: str, dest_path: str) -> None:
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_thread_count=1",
        "-o",
        "GSUtil:sliced_object_download_threshold=150",
        "-m",
        "cp",
        "-r",
        src_path,
        dest_path,
    ]
    print(f"Downloading from GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def gs_folder_exists(path: str) -> bool:
    cmd = ["gsutil", "ls", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(f"GS stat command: {cmd}")
    # print(f"GS stat stdout: {stdout}")
    # print(f"GS stat stderr: {stderr}")
    if process.returncode == 0:
        return True
    else:
        return False


def upload_to_gs_bucket(src_path: str, dest_path: str) -> None:
    cmd = ["gsutil", "-o", "GSUtil:parallel_composite_upload_threshold=150M", "cp", "-r", src_path, dest_path]
    print(f"Copying model to GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def sync_gs_bucket(src_path: str, dest_path: str) -> None:
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_composite_upload_threshold=150M",
        "-m",
        "rsync",
        "-r",
        "-d",
        src_path,
        dest_path,
    ]
    print(f"Copying model to GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def download_latest_checkpoint_from_gs(gs_checkpoint_state_dir: str, checkpoint_state_dir: str) -> None:
    """Download the latest checkpoint from GCS and update the latest file."""
    if gs_folder_exists(gs_checkpoint_state_dir):
        os.makedirs(checkpoint_state_dir, exist_ok=True)
        print(f"Downloading model checkpoint from GCS to {checkpoint_state_dir}")
        sync_gs_bucket(gs_checkpoint_state_dir, checkpoint_state_dir)


def launch_ai2_evals_on_weka(
    path: str,
    leaderboard_name: str,
    oe_eval_max_length: Optional[int] = None,
    wandb_url: Optional[str] = None,
    training_step: Optional[int] = None,
    oe_eval_tasks: Optional[List[str]] = None,
    stop_strings: Optional[List[str]] = None,
    gs_bucket_path: Optional[str] = None,
    eval_priority: Optional[str] = "normal",
) -> None:
    weka_cluster = "ai2/saturn-cirrascale ai2/neptune-cirrascale"
    gcp_cluster = "ai2/augusta-google-1"
    cluster = weka_cluster if gs_bucket_path is None else gcp_cluster
    beaker_users = get_beaker_whoami()

    if gs_bucket_path is not None:
        if beaker_users is not None:
            gs_saved_path = f"{gs_bucket_path}/{beaker_users}/{path}"
        else:
            gs_saved_path = f"{gs_bucket_path}/{path}"
        # save the model to the gs bucket first
        # TODO: use upload_to_gs_bucket instead
        gs_command = f"""gsutil \\
            -o "GSUtil:parallel_composite_upload_threshold=150M" \\
            cp -r {path} \\
            {gs_saved_path}"""
        print(f"Copying model to GS bucket with command: {gs_command}")
        process = subprocess.Popen(["bash", "-c", gs_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(f"GS bucket copy stdout:\n{stdout.decode()}")
        print(f"GS bucket copy stderr:\n{stderr.decode()}")
        print(f"GS bucket copy process return code: {process.returncode}")

        # Update path to use the GS bucket path for evaluation
        path = gs_saved_path

    command = f"""\
python scripts/submit_eval_jobs.py \
--model_name {leaderboard_name} \
--location {path} \
--cluster {cluster} \
--is_tuned \
--workspace "tulu-3-results" \
--priority {eval_priority} \
--preemptible \
--use_hf_tokenizer_template \
--run_oe_eval_experiments \
--skip_oi_evals"""
    if wandb_url is not None:
        command += f" --run_id {wandb_url}"
    if oe_eval_max_length is not None:
        command += f" --oe_eval_max_length {oe_eval_max_length}"
    if training_step is not None:
        command += f" --step {training_step}"
    if cluster == weka_cluster:
        command += " --evaluate_on_weka"
    if oe_eval_tasks is not None:
        command += f" --oe_eval_tasks {','.join(oe_eval_tasks)}"
    if stop_strings is not None:
        command += f" --oe_eval_stop_sequences '{','.join(stop_strings)}'"
    print(f"Launching eval jobs with command: {command}")
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
    print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
    print(f"Submit jobs after model training is finished - process return code: {process.returncode}")


# ----------------------------------------------------------------------------
# HF utilities


def retry_on_exception(max_attempts=4, delay=1, backoff=2):
    """
    Retry a function on exception. Useful for HF API calls that may fail due to
    network issues. E.g., https://beaker.org/ex/01J69P87HJQQ7X5DXE1CPWF974
    `huggingface_hub.utils._errors.HfHubHTTPError: 429 Client Error`

    We can test it with the following code.
    @retry_on_exception(max_attempts=4, delay=1, backoff=2)
    def test():
        raise Exception("Test exception")

    test()
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            local_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed. Retrying in {local_delay} seconds...")
                    time.sleep(local_delay)
                    local_delay *= backoff
            return None

        return wrapper

    return decorator


@retry_on_exception()
@functools.lru_cache(maxsize=1)
def maybe_use_ai2_wandb_entity() -> Optional[str]:
    """Ai2 internal logic: try use the ai2-llm team if possible. Should not affect external users."""
    import wandb

    wandb.login()
    api = wandb.Api()
    current_user = api.viewer
    teams = current_user.teams
    if "ai2-llm" in teams:
        return "ai2-llm"
    else:
        return None


@retry_on_exception()
@functools.lru_cache(maxsize=1)
def hf_whoami() -> List[str]:
    return HfApi().whoami()


@functools.lru_cache(maxsize=1)
def maybe_use_ai2_hf_entity() -> Optional[str]:
    """Ai2 internal logic: try use the allenai entity if possible. Should not affect external users."""
    orgs = hf_whoami()
    orgs = [item["name"] for item in orgs["orgs"]]
    if "allenai" in orgs:
        return "allenai"
    else:
        return None


@retry_on_exception()
def upload_metadata_to_hf(metadata_dict, filename, hf_dataset_name, hf_dataset_save_dir):
    # upload a random dict to HF. Originally for uploading metadata to HF
    # about a model for leaderboard displays.
    with open("tmp.json", "w") as f:
        json.dump(metadata_dict, f)
    api = HfApi()
    api.upload_file(
        path_or_fileobj="tmp.json",
        path_in_repo=f"{hf_dataset_save_dir}/{filename}",
        repo_id=hf_dataset_name,
        repo_type="dataset",
    )
    os.remove("tmp.json")


# ----------------------------------------------------------------------------
# Ray utilities
# Taken from https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero
def get_train_ds_config(
    offload,
    adam_offload=False,
    stage=0,
    bf16=True,
    max_norm=1.0,
    zpg=8,
    grad_accum_dtype=None,
    disable_trace_cache=False,
):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {"device": "cpu" if adam_offload else "none", "pin_memory": True},
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO++
        "zero_hpz_partition_size": zpg,
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0

    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "gradient_clipping": max_norm,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "data_types": {"grad_accum_dtype": grad_accum_dtype if grad_accum_dtype else "fp32"},
    }


def get_eval_ds_config(offload, stage=0, bf16=True):
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {"device": "cpu" if offload else "none", "pin_memory": True},
    }
    return {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }


def get_optimizer_grouped_parameters(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_name_list=["bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"],
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]


def get_ray_address() -> Optional[str]:
    """Get the Ray address from the environment variable."""
    return os.environ.get("RAY_ADDRESS")


_SET_AFFINITY = False


class RayProcess:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.world_size = world_size
        self.rank = rank
        self.local_rank = local_rank
        self.master_addr = master_addr if master_addr else self.get_current_node_ip()
        self.master_port = master_port if master_port else self.get_free_port()
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["RANK"] = str(self.rank)
        # NOTE: Ray will automatically set the CUDA_VISIBLE_DEVICES
        # environment variable for each actor, so always set device to 0
        # os.environ["LOCAL_RANK"] = str(self._local_rank)
        os.environ["LOCAL_RANK"] = "0"
        random.seed(self.rank)
        np.random.seed(self.rank)
        torch.manual_seed(self.rank)

    @staticmethod
    def get_current_node_ip():
        address = ray._private.services.get_node_ip_address()
        # strip ipv6 address
        return address.strip("[]")

    @staticmethod
    def get_free_port():
        with socket.socket() as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]

    def get_master_addr_port(self):
        return self.master_addr, self.master_port

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def _set_numa_affinity(self, rank):
        def local_rank_to_real_gpu_id(local_rank):
            cuda_visible_devices = [
                int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
            ]
            return cuda_visible_devices[local_rank]

        rank = local_rank_to_real_gpu_id(rank)

        global _SET_AFFINITY
        if _SET_AFFINITY:
            return

        from ctypes.util import find_library

        class bitmask_t(Structure):
            _fields_ = [("size", c_ulong), ("maskp", POINTER(c_ulong))]

        LIBNUMA = CDLL(find_library("numa"))
        LIBNUMA.numa_parse_nodestring.argtypes = [c_char_p]
        LIBNUMA.numa_parse_nodestring.restype = POINTER(bitmask_t)
        LIBNUMA.numa_run_on_node_mask.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_run_on_node_mask.restype = c_int
        LIBNUMA.numa_set_membind.argtypes = [POINTER(bitmask_t)]
        LIBNUMA.numa_set_membind.restype = c_void_p
        LIBNUMA.numa_num_configured_nodes.argtypes = []
        LIBNUMA.numa_num_configured_nodes.restype = c_int

        def numa_bind(nid: int):
            bitmask = LIBNUMA.numa_parse_nodestring(bytes(str(nid), "ascii"))
            LIBNUMA.numa_run_on_node_mask(bitmask)
            LIBNUMA.numa_set_membind(bitmask)

        numa_nodes = LIBNUMA.numa_num_configured_nodes()
        num_gpu_pre_numa_node = 8 // numa_nodes
        numa_bind(self.local_rank // num_gpu_pre_numa_node)
        _SET_AFFINITY = True

    def offload_to_cpu(self, model, pin_memory=True, non_blocking=True):
        """This function guaratees the memory are all released (only torch context cache <100M will remain)."""
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())

        if model.zero_optimization_stage() == 3:
            from deepspeed.runtime.zero.offload_config import OffloadStateTypeEnum

            model.optimizer.offload_states(
                include=[
                    OffloadStateTypeEnum.optim_states,
                    OffloadStateTypeEnum.contiguous_grad_buffer,
                    OffloadStateTypeEnum.hp_params,
                    # OffloadStateTypeEnum.lp_grads,
                    # OffloadStateTypeEnum.lp_params, # dangerous
                ],
                device=OffloadDeviceEnum.cpu,
                pin_memory=pin_memory,
                non_blocking=non_blocking,
            )
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage 2 is not supported yet")

    def backload_to_gpu(self, model, non_blocking=True):
        # NOTE: this function reloads the weights, ensuring the calculation
        if model.zero_optimization_stage() == 3:
            model.reload_states(non_blocking=non_blocking)
            torch.cuda.synchronize()
            return

        raise NotImplementedError("Zero stage 2 is not supported yet")


def extract_user_query(conversation: str, chat_template_name: str = None) -> str:
    # Define the regex pattern
    # if chat_template_name == "tulu_thinker":
    #     pattern = r"<answer>.*?</answer>\.\n\nUser: (.*?)\nAssistant: <think>"
    # elif chat_template_name == "tulu_thinker_r1_style":
    #     pattern = r"<answer>.*?</answer>\.\n\nUser: (.*?)\n<|assistant|>\n<think>"
    # else:
    #     # runtime error, the template is not supported
    #     raise ValueError(f"Can not extract user query for template {chat_template_name}.")
    pattern = r"\n\n<\|user\|>\n(.*?)\n<\|assistant\|>\n<think>"

    match = re.search(pattern, conversation, re.DOTALL)
    # Return the captured group if found, else return None
    return match.group(1).strip() if match else None


def extract_final_answer(prediction: str) -> str:
    """
    Extract the substring between <answer> and </answer>.
    If no match is found, extract the substring after </think>.
    If neither condition matches, clean the prediction by removing the <|assistant|> tag.
    If none of the above applies, return the original string.

    Args:
        prediction (str): The input string.

    Returns:
        str: The extracted substring or the cleaned/original string.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    if cleaned != prediction:
        return cleaned.strip()

    return prediction
