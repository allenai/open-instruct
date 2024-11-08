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

import dataclasses
import functools
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any, List, NewType, Optional, Tuple, Union

import requests
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
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["messages"] = messages
    return example


def convert_metamath_qa_to_messages(example):
    """
    Convert a query-response pair to a list of messages.
    e.g. meta-math/MetaMathQA"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
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
                    if base_type == bool:
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
    return os.path.join(folder, max(checkpoints, key=lambda x: x.split("_")[-1]))


def get_last_checkpoint_path(args, incomplete: bool = False) -> str:
    # if output already exists and user does not allow overwriting, resume from there.
    # otherwise, resume if the user specifies a checkpoint.
    # else, start from scratch.
    # if incomplete is true, include folders without "COMPLETE" in the folder.
    last_checkpoint_path = None
    if args.output_dir and os.path.isdir(args.output_dir) and not args.overwrite_output_dir:
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
    if len(checkpoints) > keep_last_n_checkpoints:
        for checkpoint in checkpoints[: len(checkpoints) - keep_last_n_checkpoints]:
            logger.info(f"Removing checkpoint {checkpoint}")
            shutil.rmtree(os.path.join(output_dir, checkpoint))
    logger.info("Remaining files:" + str(os.listdir(output_dir)))


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
def maybe_use_ai2_hf_entity() -> Optional[str]:
    """Ai2 internal logic: try use the allenai entity if possible. Should not affect external users."""
    orgs = HfApi().whoami()
    orgs = [item["name"] for item in orgs["orgs"]]
    if "allenai" in orgs:
        return "allenai"
    else:
        return None


def submit_beaker_eval_jobs(
    model_name: str,
    location: str,
    hf_repo_revision: str = "",
    workspace: str = "tulu-3-results",
    beaker_image: str = "nathanl/open_instruct_auto",
    upload_to_hf: str = "allenai/tulu-3-evals",
    run_oe_eval_experiments: bool = False,
    run_safety_evaluations: bool = False,
    skip_oi_evals: bool = False,
) -> None:
    command = f"""
    python scripts/submit_eval_jobs.py \
        --model_name {model_name} \
        --location {location} \
        --is_tuned \
        --workspace {workspace} \
        --preemptible \
        --use_hf_tokenizer_template \
        --beaker_image {beaker_image} \
    """
    if len(hf_repo_revision) > 0:
        command += f" --hf_revision {hf_repo_revision}"
    if len(upload_to_hf) > 0:
        command += f" --upload_to_hf {upload_to_hf}"
    if run_oe_eval_experiments:
        command += " --run_oe_eval_experiments"
    if run_safety_evaluations:
        command += " --run_safety_evaluations"
    if skip_oi_evals:
        command += " --skip_oi_evals"

    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    print(f"Beaker evaluation jobs: Stdout:\n{stdout.decode()}")
    print(f"Beaker evaluation jobs: Stderr:\n{stderr.decode()}")
    print(f"Beaker evaluation jobs: process return code: {process.returncode}")


@retry_on_exception()
def upload_metadata_to_hf(
    metadata_dict,
    filename,
    hf_dataset_name,
    hf_dataset_save_dir,
):
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
