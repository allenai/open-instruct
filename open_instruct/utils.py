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

# We need to set NCCL_CUMEM_ENABLE=0 for performance reasons; see:
# https://github.com/vllm-project/vllm/issues/5723#issuecomment-2554389656
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
import math
import multiprocessing as mp
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent import futures
from ctypes import CDLL, POINTER, Structure, c_char_p, c_int, c_ulong, c_void_p
from dataclasses import dataclass
from multiprocessing import resource_tracker as _rt
from typing import Any, NewType

import beaker
import numpy as np
import ray
import requests
import torch
import vllm.config
from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from dateutil import parser
from huggingface_hub import HfApi
from ray.util import state as ray_state
from rich.pretty import pprint
from tqdm import tqdm
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, AutoConfig, HfArgumentParser
from transformers.integrations import HfDeepSpeedConfig

from open_instruct import logger_utils

WEKA_CLUSTERS = ["ai2/jupiter", "ai2/saturn", "ai2/titan", "ai2/neptune", "ai2/ceres", "ai2/triton", "ai2/rhea"]
GCP_CLUSTERS = ["ai2/augusta"]
INTERCONNECT_CLUSTERS = ["ai2/jupiter", "ai2/ceres", "ai2/titan", "ai2/augusta"]

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DISK_USAGE_WARNING_THRESHOLD = 0.85
CLOUD_PATH_PREFIXES = ("gs://", "s3://", "az://", "hdfs://", "/filestore")

logger = logger_utils.setup_logger(__name__)

DataClassType = NewType("DataClassType", Any)


def warn_if_low_disk_space(
    path: str, *, threshold: float = DISK_USAGE_WARNING_THRESHOLD, send_slack_alerts: bool = False
) -> None:
    """Warns when disk usage exceeds the provided threshold.

    Args:
        path: Filesystem path to check disk usage for.
        threshold: Usage ratio (0.0-1.0) above which to warn.
        send_slack_alerts: Whether to also send a Slack alert when warning.
    """
    if path.startswith(CLOUD_PATH_PREFIXES):
        return

    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        logger.warning(f"Skipping disk usage check for {path}, encountered OS error: {e}")
        return

    if usage.total == 0:
        return

    used_ratio = usage.used / usage.total
    if used_ratio >= threshold:
        used_percent = used_ratio * 100
        free_gib = usage.free / (1024**3)
        total_gib = usage.total / (1024**3)
        warning_message = (
            f"Disk usage near capacity for {path}: {used_percent:.1f}% used "
            f"({free_gib:.1f} GiB free of {total_gib:.1f} GiB). Checkpointing may fail."
        )
        logger.warning(warning_message)
        if send_slack_alerts:
            send_slack_message(f"{warning_message}")


class MetricsTracker:
    """A simple class to preallocate all metrics in an array
    so we can do only one allreduce operation to get the metrics mean"""

    def __init__(self, max_metrics: int = 32, device: str = "cuda"):
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def _maybe_register_metric(self, name: str) -> int:
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1
        return self.names2idx[name]

    def __getitem__(self, name: str) -> torch.Tensor:
        idx = self._maybe_register_metric(name)
        return self.metrics[idx]

    def __setitem__(self, name: str, value):
        idx = self._maybe_register_metric(name)
        self.metrics[idx] = value

    def get_metrics_list(self) -> dict[str, float]:
        # Convert to Python floats for logging systems (wandb, tensorboard)
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}


def max_num_processes() -> int:
    """Returns a reasonable default number of processes to run for multiprocessing."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count() or 1


def repeat_each(seq, k):
    """Repeat each element in a sequence k times."""
    return [item for item in seq for _ in range(k)]


def ray_get_with_progress(
    ray_refs: list[ray.ObjectRef], desc: str = "Processing", enable: bool = True, timeout: float | None = None
):
    """Execute ray.get() with a progress bar using futures and collect timings.

    Args:
        ray_refs: List of ray object references
        desc: Description for the progress bar
        enable: Whether to show the progress bar (default: True)
        timeout: Optional timeout in seconds for all operations to complete

    Returns:
        (results, completion_times)
        - results: List of results in the same order as ray_refs
        - completion_times: time from function start until each ref completed (seconds), aligned to ray_refs

    Raises:
        TimeoutError: If timeout is specified and operations don't complete in time
    """
    t0 = time.perf_counter()

    ray_futures = [ref.future() for ref in ray_refs]
    fut_to_idx = {f: i for i, f in enumerate(ray_futures)}

    results = [None] * len(ray_refs)
    completion_times = [None] * len(ray_refs)

    futures_iter = futures.as_completed(ray_futures, timeout=timeout)
    if enable:
        futures_iter = tqdm(futures_iter, total=len(ray_futures), desc=desc, bar_format="{l_bar}{bar}{r_bar}\n")

    try:
        for future in futures_iter:
            idx = fut_to_idx[future]
            results[idx] = future.result()
            completion_times[idx] = time.perf_counter() - t0
    except TimeoutError as e:
        raise TimeoutError(f"{desc} failed.") from e

    return results, completion_times


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
    dataset_mixer: dict | list,
    splits: list[str] | None = None,
    configs: list[str] | None = None,
    columns_to_keep: list[str] | None = None,
    shuffle: bool = True,
    save_data_dir: str | None = None,
    need_columns: list[str] | None = None,
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
            value = float(dataset_mixer[i + 1]) if "." in dataset_mixer[i + 1] else int(dataset_mixer[i + 1])
            mixer_dict[dataset_mixer[i]] = value
            i += 2
        dataset_mixer = mixer_dict

    splits = ["train", "test"] if splits is None else splits
    configs = configs if configs else [None] * len(dataset_mixer)
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
                dataset = load_dataset("json", data_files=ds, split=split, num_proc=max_num_processes())
            elif ds.endswith(".parquet"):
                dataset = load_dataset("parquet", data_files=ds, split=split, num_proc=max_num_processes())
            else:
                try:
                    # Try first if dataset on a Hub repo
                    dataset = load_dataset(ds, ds_config, split=split, num_proc=max_num_processes())
                except DatasetGenerationError:
                    # If not, check local dataset
                    dataset = load_from_disk(os.path.join(ds, split))

            # shuffle dataset if set
            if shuffle:
                dataset = dataset.shuffle(seed=42)

            # assert that needed columns are present
            if need_columns and not all(col in dataset.column_names for col in need_columns):
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
        if len(raw_train_datasets) > 0 and "id" in raw_datasets["train"].column_names:
            raw_datasets["train"] = raw_datasets["train"].remove_columns("id")
        if len(raw_val_datasets) > 0 and "id" in raw_datasets["test"].column_names:
            raw_datasets["test"] = raw_datasets["test"].remove_columns("id")

    return raw_datasets


def combine_dataset(
    dataset_mixer: dict | list,
    splits: list[str],
    configs: list[str] | None = None,
    columns_to_keep: list[str] | None = None,
    shuffle: bool = False,
    save_data_dir: str | None = None,
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
            value = float(dataset_mixer[i + 1]) if "." in dataset_mixer[i + 1] else int(dataset_mixer[i + 1])
            mixer_dict[dataset_mixer[i]] = value
            i += 2
        dataset_mixer = mixer_dict

    if any(frac_or_samples < 0 for frac_or_samples in dataset_mixer.values()):
        raise ValueError("Dataset fractions / lengths cannot be negative.")

    configs = configs if configs else [None] * len(dataset_mixer)
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
            dataset = load_dataset("json", data_files=ds, split=split, num_proc=max_num_processes())
        else:
            try:
                # Try first if dataset on a Hub repo
                dataset = load_dataset(ds, ds_config, split=split, num_proc=max_num_processes())
            except DatasetGenerationError:
                # If not, check local dataset
                dataset = load_from_disk(os.path.join(ds, split))

        # shuffle dataset if set
        if shuffle:
            dataset = dataset.shuffle(seed=42)

        # select a fraction of the dataset
        samples = int(frac_or_samples) if frac_or_samples > 1.0 else int(frac_or_samples * len(dataset))
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

    if not keep_ids and "id" in datasets.column_names:
        datasets = datasets.remove_columns("id")

    return datasets


# ----------------------------------------------------------------------------
# Arguments utilities
class ArgumentParserPlus(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: list[str] | None = None) -> list[dataclass]:
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

                    if base_type == list[str]:
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

    def parse(self) -> DataClassType | tuple[DataClassType]:
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
def get_wandb_tags() -> list[str]:
    """Get tags for Weights & Biases (e.g., `no-tag-404-g98dc659,pr-123,branch-main`)"""
    tags = [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t != ""]
    if "GIT_COMMIT" in os.environ:
        git_commit = os.environ["GIT_COMMIT"]
        tags.append(f"commit: {git_commit}")
        try:
            # try finding the pull request number on github
            prs = requests.get(f"https://api.github.com/search/issues?q=repo:allenai/open-instruct+is:pr+{git_commit}")
            prs.raise_for_status()
            prs = prs.json()
            pr = prs["items"][0]
            tags.append(f"pr: {pr['number']}")
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to get PR number from GitHub API: {e}.")
    if "GIT_BRANCH" in os.environ:
        tags.append(f"branch: {os.environ['GIT_BRANCH']}")
    tags = [tag[:64] for tag in tags]
    return tags


# ----------------------------------------------------------------------------
# Check pointing utilities
def get_last_checkpoint(folder: str, incomplete: bool = False) -> str | None:
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
    beaker_node_hostname: list[str] | None = None
    beaker_experiment_url: list[str] | None = None
    beaker_dataset_ids: list[str] | None = None
    beaker_dataset_id_urls: list[str] | None = None


def is_beaker_job() -> bool:
    return "BEAKER_JOB_ID" in os.environ


def get_beaker_experiment_info(experiment_id: str) -> dict | None:
    get_experiment_command = f"beaker experiment get {experiment_id} --format json"
    process = subprocess.Popen(["bash", "-c", get_experiment_command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Failed to get Beaker experiment: {stderr}")
        return None
    return json.loads(stdout)[0]


def beaker_experiment_succeeded(experiment_id: str) -> bool:
    experiment = get_beaker_experiment_info(experiment_id)
    num_replicas = experiment["jobs"][0]["execution"]["spec"].get("replicas", 1)
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


def get_beaker_dataset_ids(experiment_id: str, sort=False) -> list[str] | None:
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
def get_beaker_whoami() -> str | None:
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


def format_eta(seconds: float) -> str:
    """Format ETA in a human-readable format."""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def maybe_update_beaker_description(
    current_step: int | None = None,
    total_steps: int | None = None,
    start_time: float | None = None,
    wandb_url: str | None = None,
    original_descriptions: dict[str, str] = {},  # noqa: B006
) -> None:
    """Update Beaker experiment description with training progress and/or wandb URL.

    Args:
        current_step: Current training step (for progress tracking)
        total_steps: Total number of training steps (for progress tracking)
        start_time: Training start time (from time.time()) (for progress tracking)
        wandb_url: Optional wandb URL to include
        original_descriptions: Cache of original descriptions for progress updates
    """
    if not is_beaker_job():
        return

    experiment_id = os.environ.get("BEAKER_WORKLOAD_ID")
    if not experiment_id:
        logger.warning(
            f"BEAKER_WORKLOAD_ID not found in environment. Available env vars: {', '.join(sorted([k for k in os.environ if 'BEAKER' in k]))}"
        )
        return

    try:
        client = beaker.Beaker.from_env()
    except beaker.exceptions.BeakerConfigurationError as e:
        logger.warning(f"Failed to initialize Beaker client: {e}")
        return

    try:
        # Get the workload first (experiment_id is actually BEAKER_WORKLOAD_ID)
        workload = client.workload.get(experiment_id)
        # Then get the experiment spec from the workload
        spec = client.experiment.get_spec(workload)
    except (beaker.exceptions.BeakerExperimentNotFound, ValueError):
        logger.warning(
            f"Failed to get Beaker experiment with ID: {experiment_id}"
            "This might be fine if you are e.g. running in an interactive job."
        )
        return

    if experiment_id not in original_descriptions:
        raw_description = spec.description or ""
        if "git_commit:" in raw_description:
            raw_description = raw_description.split("git_commit:")[0].strip()
        original_descriptions[experiment_id] = raw_description

    # Build description from scratch each time
    description_components = [
        original_descriptions[experiment_id],
        f"git_commit: {os.environ.get('GIT_COMMIT', 'unknown')}",
        f"git_branch: {os.environ.get('GIT_BRANCH', 'unknown')}",
    ]

    if wandb_url:
        description_components.append(wandb_url)

    if current_step is not None:
        progress_pct = (current_step / total_steps) * 100
        elapsed_time = time.perf_counter() - start_time

        if current_step >= total_steps:
            time_str = format_eta(elapsed_time)
            time_label = "finished in"
        else:
            if current_step > 0:
                time_per_step = elapsed_time / current_step
                remaining_steps = total_steps - current_step
                eta_seconds = time_per_step * remaining_steps
                time_str = format_eta(eta_seconds)
            else:
                time_str = "calculating..."
            time_label = "eta"

        progress_bar = f"[{progress_pct:.1f}% complete (step {current_step}/{total_steps}), {time_label} {time_str}]"
        description_components.append(progress_bar)
    new_description = " ".join(description_components)
    try:
        # Update the workload description using the workload object we got earlier
        client.workload.update(workload, description=new_description)
    except requests.exceptions.HTTPError as e:
        logger.warning(
            f"Failed to update Beaker description due to HTTP error: {e}"
            "Continuing without updating description - this is likely a temporary Beaker service issue"
        )


def live_subprocess_output(cmd: list[str]) -> str:
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


def download_from_gs_bucket(src_paths: list[str], dest_path: str) -> None:
    os.makedirs(dest_path, exist_ok=True)
    cmd = [
        "gsutil",
        "-o",
        "GSUtil:parallel_thread_count=1",
        "-o",
        "GSUtil:sliced_object_download_threshold=150",
        "-m",
        "cp",
        "-r",
    ]
    cmd.extend(src_paths)
    cmd.append(dest_path)
    print(f"Downloading from GS bucket with command: {cmd}")
    live_subprocess_output(cmd)


def gs_folder_exists(path: str) -> bool:
    cmd = ["gsutil", "ls", path]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    # print(f"GS stat command: {cmd}")
    # print(f"GS stat stdout: {stdout}")
    # print(f"GS stat stderr: {stderr}")
    return process.returncode == 0


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
    oe_eval_max_length: int | None = None,
    wandb_url: str | None = None,
    training_step: int | None = None,
    oe_eval_tasks: list[str] | None = None,
    stop_strings: list[str] | None = None,
    gs_bucket_path: str | None = None,
    eval_priority: str | None = "normal",
    eval_workspace: str | None = "ai2/tulu-3-results",
    beaker_image: str | None = None,
    oe_eval_gpu_multiplier: int | None = None,
) -> None:
    beaker_users = get_beaker_whoami()

    if gs_bucket_path is not None:
        cluster_str = f"--cluster {' '.join(GCP_CLUSTERS)}"
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
    else:
        cluster_str = ""
    command = f"""\
python scripts/submit_eval_jobs.py \
--model_name {leaderboard_name} \
--location {path} {cluster_str} \
--is_tuned \
--workspace {eval_workspace} \
--priority {eval_priority} \
--preemptible \
--use_hf_tokenizer_template \
--run_oe_eval_experiments \
--skip_oi_evals"""
    if wandb_url is not None:
        command += f" --run_id {wandb_url}"
        wandb_run_path = wandb_url_to_run_path(wandb_url)
        command += f" --wandb_run_path {wandb_run_path}"
    if oe_eval_max_length is not None:
        command += f" --oe_eval_max_length {oe_eval_max_length}"
    if training_step is not None:
        command += f" --step {training_step}"
    if gs_bucket_path is None:
        command += " --evaluate_on_weka"
    if oe_eval_tasks is not None:
        command += f" --oe_eval_tasks {','.join(oe_eval_tasks)}"
    if stop_strings is not None:
        command += f" --oe_eval_stop_sequences '{','.join(stop_strings)}'"
    if beaker_image is not None:
        command += f" --beaker_image {beaker_image}"
    if oe_eval_gpu_multiplier is not None:
        command += f" --gpu_multiplier {oe_eval_gpu_multiplier}"
    print(f"Launching eval jobs with command: {command}")
    process = subprocess.Popen(["bash", "-c", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(f"Submit jobs after model training is finished - Stdout:\n{stdout.decode()}")
    print(f"Submit jobs after model training is finished - Stderr:\n{stderr.decode()}")
    print(f"Submit jobs after model training is finished - process return code: {process.returncode}")


def wandb_url_to_run_path(url: str) -> str:
    """
    Convert a wandb URL to a wandb run path.

    Args:
        url (str): wandb URL in format https://wandb.ai/entity/project/runs/run_id

    Returns:
        str: wandb run path in format entity/project/run_id

    >>> wandb_url_to_run_path("https://wandb.ai/org/project/runs/runid")
    org/project/runid

    >>> wandb_url_to_run_path("https://wandb.ai/ai2-llm/open_instruct_internal/runs/5nigq0mz")
    ai2-llm/open_instruct_internal/5nigq0mz
    """
    # Remove the base URL and split by '/'
    path_parts = url.replace("https://wandb.ai/", "").split("/")

    # Extract entity, project, and run_id
    entity = path_parts[0]
    project = path_parts[1]
    run_id = path_parts[3]  # Skip 'runs' at index 2

    return f"{entity}/{project}/{run_id}"


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
def maybe_use_ai2_wandb_entity() -> str | None:
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
def hf_whoami() -> list[str]:
    return HfApi().whoami()


@functools.lru_cache(maxsize=1)
def maybe_use_ai2_hf_entity() -> str | None:
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


def get_eval_ds_config(
    offload: bool, stage: int = 0, bf16: bool = True, per_device_train_batch_size: int = 1
) -> tuple[dict[str, Any], HfDeepSpeedConfig | None]:
    """Creates a DeepSpeed configuration for evaluation.

    Args:
        offload: Whether to offload parameters to CPU.
        stage: ZeRO optimization stage. Only 0 or 3 are relevant as there's no optimizer for eval.
        bf16: Whether to enable bfloat16 precision.
        per_device_train_batch_size: Batch size per GPU.

    Returns:
        Tuple containing a Dictionary containing DeepSpeed configuration, and the actual HfDeepSpeedConfig object if stage 3 is used, else None. We need to return the HfDeepSpeedConfig object so it doesn't go out of scope as HF accelerate uses it internally via a global weakref.

    Raises:
        ValueError: If stage is not 0 or 3.
    """
    if stage not in (0, 3):
        raise ValueError(
            f"stage must be 0 or 3 for evaluation (got {stage}). 1 or 2 only differ from stage 0 by optimizer sharding, which is irrelevant for evaluation."
        )
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": "auto",
        "offload_param": {"device": "cpu" if offload else "none", "pin_memory": True},
    }
    ds_config = {
        "steps_per_print": 100,
        "zero_optimization": zero_opt_dict,
        "bf16": {"enabled": bf16},
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
    ds_config["train_micro_batch_size_per_gpu"] = per_device_train_batch_size
    ds_config["gradient_accumulation_steps"] = 1
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        # This is needed as it apparently has mysterious side effects.
        hf_config = HfDeepSpeedConfig(ds_config)
        logger.info(f"DeepSpeed config: {hf_config}")
    else:
        hf_config = None
    return ds_config, hf_config


def get_optimizer_grouped_parameters(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_name_list=("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight"),
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


def get_ray_address() -> str | None:
    """Get the Ray address from the environment variable."""
    return os.environ.get("RAY_ADDRESS")


_SET_AFFINITY = False


class RayProcess:
    def __init__(self, world_size, rank, local_rank, master_addr, master_port):
        logger_utils.setup_logger()
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
    pattern = re.compile(
        r"(?:"
        r"<\|user\|\>\n(?P<simple>.*?)\n<\|assistant\|\>\n<think>"  # template 0 (your original)
        r"|"
        r"<\|im_start\|\>user\n(?P<im>.*?)(?:\n<functions>.*?</functions>)?<\|im_end\|\>\n"  # templates 1 & 2
        r"(?=[\s\S]*?<\|im_start\|\>assistant\n<think>)"  # ensure it's the turn before <think>
        r")",
        re.DOTALL,
    )
    # Get the last user query matched (most recent user turn before assistant <think>)
    matches = list(pattern.finditer(conversation))
    if matches:
        m = matches[-1]
        user_query = (m.group("simple") or m.group("im")).strip()
    else:
        user_query = conversation

    return user_query


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


# ---- Runtime leak detection -----------------------------------------------------------------

DEFAULT_THREAD_ALLOWLIST = {
    "MainThread",
    "pytest-watcher",  # pytest
    "pydevd.",  # debugger
    "IPythonHistorySavingThread",
    "raylet_client",  # ray internal when still up during test body
}

DEFAULT_THREAD_ALLOW_PREFIXES = {
    "ThreadPoolExecutor-",  # executors create transient threads; adjust if you join them
    "ray-",  # ray internal threads
    "grpc-default-executor",  # grpc internal
}


def check_runtime_leaks(
    thread_allowlist: Iterable[str] = DEFAULT_THREAD_ALLOWLIST,
    thread_allow_prefixes: Iterable[str] = DEFAULT_THREAD_ALLOW_PREFIXES,
    include_daemon_threads: bool = False,
) -> None:
    """
    Inspect runtime state for leftovers and log any leaks immediately.
    """
    leak_logger = logging.getLogger(__name__)

    def is_allowed_thread(t):
        return (
            t.name in thread_allowlist
            or any(t.name.startswith(p) for p in thread_allow_prefixes)
            or t is threading.main_thread()
            or (not include_daemon_threads and t.daemon)
            or not t.is_alive()
        )

    bad_threads = [t for t in threading.enumerate() if not is_allowed_thread(t)]
    if bad_threads:
        leak_logger.warning("Leaked threads:")
        for t in bad_threads:
            target = getattr(t, "_target", None)
            tgt_name = getattr(target, "__name__", repr(target)) if target else "?"
            leak_logger.warning(f"  - {t.name} (alive={t.is_alive()}, daemon={t.daemon}, target={tgt_name})")

    bad_processes = [p for p in mp.active_children() if p.is_alive()]
    if bad_processes:
        leak_logger.warning("Leaked multiprocessing children:")
        for p in bad_processes:
            leak_logger.warning(f"  - PID {p.pid} alive={p.is_alive()} name={p.name}")

    if ray_state and ray and ray.is_initialized():
        ray_checks = [
            (
                "Live Ray actors:",
                ray_state.list_actors(filters=[("state", "=", "ALIVE")]),
                lambda a: f"  - {a.get('class_name')} id={a.get('actor_id')}",
            ),
            (
                "Live Ray tasks:",
                ray_state.list_tasks(filters=[("state", "=", "RUNNING")]),
                lambda t: f"  - {t.get('name')} id={t.get('task_id')}",
            ),
            (
                "Live Ray workers:",
                ray_state.list_workers(filters=[("is_alive", "=", True)]),
                lambda w: f"  - pid={w.get('pid')} id={w.get('worker_id')}",
            ),
        ]

        for header, items, formatter in ray_checks:
            if items:
                leak_logger.warning(header)
                for item in items:
                    leak_logger.warning(formatter(item))

    if _rt and hasattr(_rt, "_resource_tracker"):
        cache = getattr(_rt._resource_tracker, "_cache", {})
        for count, rtype in cache.values():
            if count > 0:
                leak_logger.warning(f"Leaked {rtype} resources: {count}")


def check_oe_eval_internal():
    """Check if oe-eval-internal is available when running in Beaker.

    Raises an error if we're running in Beaker but oe-eval-internal is not present.
    This is needed because oe-eval-internal is required for certain evaluation tasks
    but is only available internally at AI2.
    """
    # Return early if not running in Beaker
    if not os.environ.get("BEAKER_EXPERIMENT_ID"):
        return

    # We're in Beaker, check if oe-eval-internal exists
    if not os.path.exists("/stage/oe-eval-internal"):
        raise RuntimeError(
            "Running in Beaker but oe-eval-internal directory is not found. "
            "The oe-eval-internal repository is required for evaluation tasks "
            "when running in Beaker. Please ensure the Docker image was built "
            "with access to the oe-eval-internal repository."
        )


# For FLOPS, we assume bf16 and ignore sparsity.
# Memory bandwidth values are peak theoretical bandwidth.
GPU_SPECS = {
    "a100": {"flops": 312e12, "memory_size": 80e9, "memory_bandwidth": 2.0e12},  # 2.0 TB/s HBM2e (80GB variant)
    "b200": {"flops": 2250e12, "memory_size": 192e9, "memory_bandwidth": 8e12},  # 8 TB/s HBM3e
    "h100": {"flops": 990e12, "memory_size": 80e9, "memory_bandwidth": 3.35e12},  # 3.35 TB/s HBM3
    "a6000": {"flops": 155e12, "memory_size": 48e9, "memory_bandwidth": 768e9},  # 768 GB/s GDDR6
    "l40s": {"flops": 362e12, "memory_size": 48e9, "memory_bandwidth": 864e9},  # 864 GB/s GDDR6
    "pro 6000": {"flops": 503.8e12, "memory_size": 96e9, "memory_bandwidth": 1792e9},  # 1792 GB/s GDDR7
    "6000": {"flops": 728.5e12, "memory_size": 48e9, "memory_bandwidth": 960e9},  # 960 GB/s GDDR6
    # Specs from https://www.techpowerup.com/gpu-specs/geforce-rtx-4090-mobile.c3949.
    "4090 laptop": {"flops": 32.98e12, "memory_size": 24e9, "memory_bandwidth": 576e9},
}

# Conventions for FLOPs calculations (fixed; not switches)
FLOP_PER_MAC = 2
# Approximate softmax cost per attention score:
# ~4 scalar ops/score: exp + subtract max (stabilization) + sum + divide.
SOFTMAX_FLOPS_PER_SCORE = 4


@dataclasses.dataclass
class ModelDims:
    num_layers: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attn_heads: int
    head_dim: int
    num_kv_heads: int | None = None
    num_params: int | None = None
    device_name: str | None = None
    sliding_window: int | None = None
    num_sliding_window_layers: int = 0

    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_attn_heads

        self.num_params = self.num_params or self._calculate_num_params()

        if self.device_name is None and torch.cuda.is_available():
            self.device_name = get_device_name(torch.cuda.get_device_name(0))

        assert self.hidden_size % self.num_attn_heads == 0, "hidden_size must be divisible by num_attn_heads"
        assert self.num_attn_heads % self.num_kv_heads == 0, (
            "num_attn_heads must be divisible by num_kv_heads (GQA/MQA)"
        )
        assert self.num_sliding_window_layers <= self.num_layers, (
            f"num_sliding_window_layers ({self.num_sliding_window_layers}) cannot exceed num_layers ({self.num_layers})"
        )

    def _calculate_num_params(self) -> int:
        embedding_params = self.vocab_size * self.hidden_size

        q_params = self.hidden_size * (self.num_attn_heads * self.head_dim)
        kv_params = self.hidden_size * (self.num_kv_heads * self.head_dim) * 2
        o_params = (self.num_attn_heads * self.head_dim) * self.hidden_size
        mlp_up_params = self.hidden_size * self.intermediate_size * 2
        mlp_down_params = self.intermediate_size * self.hidden_size

        per_layer_params = q_params + kv_params + o_params + mlp_up_params + mlp_down_params
        layer_params = self.num_layers * per_layer_params

        lm_head_params = self.vocab_size * self.hidden_size

        return embedding_params + layer_params + lm_head_params

    @classmethod
    def from_vllm_config(cls, vllm_config: vllm.config.VllmConfig) -> "ModelDims":
        """Create ModelDims from a vLLM config object."""
        model_config = vllm_config.model_config
        hidden_size = model_config.get_hidden_size()

        # Try to get intermediate_size, default to 4x hidden_size if not present
        intermediate_size = getattr(model_config.hf_text_config, "intermediate_size", 4 * hidden_size)

        sliding_window = getattr(model_config.hf_text_config, "sliding_window", None)
        num_layers = model_config.get_num_layers(vllm_config.parallel_config)
        num_sliding_window_layers = 0

        if sliding_window is not None:
            layer_types = getattr(model_config.hf_text_config, "layer_types", None)
            if layer_types is not None:
                num_sliding_window_layers = layer_types.count("sliding_attention")
            else:
                # If "layer_types" is None, then we assume all layers are sliding layers.
                num_sliding_window_layers = num_layers

        return cls(
            num_layers=num_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=model_config.get_vocab_size(),
            num_attn_heads=model_config.hf_text_config.num_attention_heads,
            num_kv_heads=model_config.hf_text_config.num_key_value_heads,
            head_dim=model_config.get_head_size(),
            sliding_window=sliding_window,
            num_sliding_window_layers=num_sliding_window_layers,
            device_name=get_device_name(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
        )

    @classmethod
    def from_hf_config(cls, model_name_or_path: str) -> "ModelDims":
        """Create ModelDims from a HuggingFace model name or path."""
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        hidden_size = config.hidden_size
        intermediate_size = getattr(config, "intermediate_size", 4 * hidden_size)
        sliding_window = getattr(config, "sliding_window", None)
        num_sliding_window_layers = 0
        if sliding_window is not None:
            layer_types = getattr(config, "layer_types", None)
            if layer_types is not None:
                num_sliding_window_layers = layer_types.count("sliding_attention")
            else:
                num_sliding_window_layers = config.num_hidden_layers
        head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
        return cls(
            num_layers=config.num_hidden_layers,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=config.vocab_size,
            num_attn_heads=config.num_attention_heads,
            num_kv_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
            head_dim=head_dim,
            sliding_window=sliding_window,
            num_sliding_window_layers=num_sliding_window_layers,
            device_name=get_device_name(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None,
        )

    @property
    def device_flops(self) -> float:
        assert self.device_name is not None, "device_name must be set"
        assert self.device_name in GPU_SPECS, f"Unknown device: {self.device_name}"
        return GPU_SPECS[self.device_name]["flops"]

    @property
    def device_memory_bandwidth(self) -> float:
        assert self.device_name is not None, "device_name must be set"
        assert self.device_name in GPU_SPECS, f"Unknown device: {self.device_name}"
        return GPU_SPECS[self.device_name]["memory_bandwidth"]

    def attn_flops(self, query_len: int, kv_len: int, sliding_window: int | None = None) -> int:
        """FLOPs for one layer of self-attention given query_len and kv_len.

        Assumptions:
          - 1 MAC = 2 FLOPs (FLOP_PER_MAC).
          - Efficient GQA/MQA K/V projections with width = num_kv_heads * head_dim.
          - Softmax ≈ 4 FLOPs per score (see SOFTMAX_FLOPS_PER_SCORE).
          - LayerNorms and minor ops ignored (dominated by matmuls).
        """
        d = self.head_dim
        mul = FLOP_PER_MAC

        q_dim = self.num_attn_heads * d
        kv_dim = self.num_kv_heads * d

        kv_len = min(kv_len, sliding_window or float("inf"))

        # Projections for the query_len new tokens
        q_proj = mul * query_len * self.hidden_size * q_dim
        kv_proj = mul * 2 * query_len * self.hidden_size * kv_dim  # GQA/MQA

        # Scores and attention-weighted values
        qk = mul * self.num_attn_heads * query_len * kv_len * d
        softmax = SOFTMAX_FLOPS_PER_SCORE * self.num_attn_heads * query_len * kv_len
        av = mul * self.num_attn_heads * query_len * kv_len * d

        # Output projection
        out_proj = mul * query_len * q_dim * self.hidden_size

        return q_proj + kv_proj + qk + softmax + av + out_proj

    def mlp_flops(self, seq_len: int) -> int:
        """Two matmuls dominate; activation cost under-counted on purpose."""
        mul = FLOP_PER_MAC
        first = mul * seq_len * self.hidden_size * (self.intermediate_size * 2)  # times 2 due to SwiGLU
        act = seq_len * self.intermediate_size  # under-counted on purpose
        second = mul * seq_len * self.intermediate_size * self.hidden_size
        return first + act + second

    def prefill_flops(self, prompt_lengths: list[int]) -> int:
        """Prefill builds the KV cache; logits are computed once after each prompt."""
        num_full_attn_layers = self.num_layers - self.num_sliding_window_layers
        num_sliding_layers = self.num_sliding_window_layers

        total = 0
        for L in prompt_lengths:
            if num_full_attn_layers > 0:
                total += num_full_attn_layers * (self.attn_flops(L, L, sliding_window=None) + self.mlp_flops(L))

            if num_sliding_layers > 0:
                total += num_sliding_layers * (
                    self.attn_flops(L, L, sliding_window=self.sliding_window) + self.mlp_flops(L)
                )

            # Always include a single LM head after prefill (next-token logits)
            total += FLOP_PER_MAC * self.hidden_size * self.vocab_size

        return total

    def decode_flops(self, prompt_lengths: list[int], response_lengths: list[int], samples_per_prompt: int = 1) -> int:
        """Decode/generation FLOPs.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt

        Embedding lookups are ignored by design.
        """
        assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
            f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
        )

        num_full_attn_layers = self.num_layers - self.num_sliding_window_layers
        num_sliding_layers = self.num_sliding_window_layers

        total = 0
        response_idx = 0
        for P in prompt_lengths:
            # Process all samples for this prompt
            for _ in range(samples_per_prompt):
                R = response_lengths[response_idx]
                total += R * self.num_layers * self.mlp_flops(seq_len=1)
                for t in range(R):
                    kv_len = P + t + 1  # prompt + generated so far + current
                    if num_full_attn_layers > 0:
                        total += num_full_attn_layers * self.attn_flops(
                            query_len=1, kv_len=kv_len, sliding_window=None
                        )
                    if num_sliding_layers > 0:
                        total += num_sliding_layers * self.attn_flops(
                            query_len=1, kv_len=kv_len, sliding_window=self.sliding_window
                        )
                total += R * FLOP_PER_MAC * self.hidden_size * self.vocab_size
                response_idx += 1
        return total

    def flops(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        is_training: bool = False,
    ) -> int:
        """Total FLOPs for prefill and (optionally) decode.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            is_training: If True, multiply FLOPs by 3 to account for forward and backward passes
        """
        total = self.prefill_flops(prompt_lengths)
        if response_lengths is not None:
            total += self.decode_flops(prompt_lengths, response_lengths, samples_per_prompt)
        if is_training:
            # Training includes forward pass (1x) + backward pass (2x)
            total *= 3
        return total

    def weight_memory_bytes(self, num_tokens: int, dtype_bytes: int = 2) -> int:
        """Memory bytes for reading model weights for a given number of tokens.

        Args:
            num_tokens: Number of tokens to process
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total bytes for weight reads across all layers
        """
        hidden_q = self.num_attn_heads * self.head_dim
        hidden_kv = self.num_kv_heads * self.head_dim

        # Per-layer weight params (Q, K, V, O, MLP up, MLP down)
        w_q = self.hidden_size * hidden_q
        w_k = self.hidden_size * hidden_kv
        w_v = self.hidden_size * hidden_kv
        w_o = hidden_q * self.hidden_size
        w_up = self.hidden_size * (self.intermediate_size * 2)  # times 2 due to SwiGLU
        w_dn = self.intermediate_size * self.hidden_size

        per_layer_weight_bytes = (w_q + w_k + w_v + w_o + w_up + w_dn) * dtype_bytes
        return self.num_layers * num_tokens * per_layer_weight_bytes

    def kv_cache_write_bytes(self, num_tokens: int, dtype_bytes: int = 2) -> int:
        """Memory bytes for writing KV cache for a given number of tokens.

        Args:
            num_tokens: Number of tokens being cached
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total bytes for KV cache writes across all layers
        """
        # 2x for K and V
        kv_write_bytes_per_token = 2 * self.num_kv_heads * self.head_dim * dtype_bytes
        return self.num_layers * num_tokens * kv_write_bytes_per_token

    def kv_cache_read_bytes(
        self, prompt_lengths: list[int], response_lengths: list[int], samples_per_prompt: int = 1, dtype_bytes: int = 2
    ) -> int:
        """Memory bytes for reading KV cache during decode.

        For each new token generated, we read all previous tokens' KV cache.
        When generating multiple samples per prompt, the prompt KV cache is shared.

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total bytes for KV cache reads during decode
        """
        assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
            f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
        )

        num_full_attn_layers = self.num_layers - self.num_sliding_window_layers
        num_sliding_layers = self.num_sliding_window_layers

        # For batched sampling with shared prompt KV cache:
        # - Prompt KV is read once per new token position across ALL samples (not per sample)
        # - Each sample has its own KV for generated tokens
        kv_read_terms = 0
        response_idx = 0

        for P in prompt_lengths:
            # For this prompt, collect all response lengths
            prompt_responses = []
            for _ in range(samples_per_prompt):
                prompt_responses.append(response_lengths[response_idx])
                response_idx += 1

            # Prompt KV reads: In synchronized batch generation with vLLM n>1,
            # the prompt KV cache is stored once but each sample reads it independently.
            # At each decoding position, each sample reads the prompt KV cache.
            # Number of positions = max response length (all generate synchronously).
            max_response_length = max(prompt_responses) if prompt_responses else 0
            # Each of the samples_per_prompt samples reads prompt KV at each position
            kv_read_terms += max_response_length * samples_per_prompt * P * num_full_attn_layers

            # Per-sample generated KV reads: Each sample reads its own previously generated tokens
            for R in prompt_responses:
                # Each token in this sample reads its previously generated tokens
                kv_read_terms += num_full_attn_layers * R * (R - 1) // 2
                if num_sliding_layers > 0:
                    # ... unless we have a sliding window, at which point we cap the max tokens to read.
                    # Note that we also account for the prompt KV values here as well.
                    kv_read_terms += num_sliding_layers * sum(min(P + t, self.sliding_window) for t in range(R))
        # 2x for K and V
        kv_bytes_per_token = 2 * self.num_kv_heads * self.head_dim * dtype_bytes
        return kv_bytes_per_token * kv_read_terms

    def prefill_memory_bytes(self, prompt_lengths: list[int], dtype_bytes: int = 2) -> int:
        """Memory bytes for prefill phase.

        During prefill:
        - Read weights once per prefill operation
        - Write KV cache for each token

        Args:
            prompt_lengths: List of prompt lengths
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total memory bytes for prefill
        """
        num_prefill_ops = 1
        weight_bytes = self.weight_memory_bytes(num_prefill_ops, dtype_bytes)
        total_prefill_tokens = sum(prompt_lengths)
        kv_write_bytes = self.kv_cache_write_bytes(total_prefill_tokens, dtype_bytes)
        return weight_bytes + kv_write_bytes

    def decode_memory_bytes(
        self, prompt_lengths: list[int], response_lengths: list[int], samples_per_prompt: int = 1, dtype_bytes: int = 2
    ) -> int:
        """Memory bytes for decode/generation phase.

        During decode:
        - Read weights for each new token position (shared across samples in batch)
        - Write KV cache for each new token
        - Read all previous KV cache for attention

        Args:
            prompt_lengths: List of prompt lengths (one per unique prompt)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Total memory bytes for decode
        """
        # In synchronized batch generation, weights are read once per position,
        # not once per token. With multiple samples per prompt generating in parallel,
        # we only need to read weights for the number of unique positions.
        unique_positions = 0
        response_idx = 0
        for _ in prompt_lengths:
            # Get response lengths for this prompt's samples
            prompt_responses = response_lengths[response_idx : response_idx + samples_per_prompt]
            response_idx += samples_per_prompt
            # In synchronized generation, all samples generate the same number of positions
            # (up to the max length among them)
            unique_positions += max(prompt_responses) if prompt_responses else 0

        weight_bytes = self.weight_memory_bytes(unique_positions, dtype_bytes)

        # KV writes happen for all tokens (each sample writes its own KV)
        total_decode_tokens = sum(response_lengths)
        kv_write_bytes = self.kv_cache_write_bytes(total_decode_tokens, dtype_bytes)

        kv_read_bytes = self.kv_cache_read_bytes(prompt_lengths, response_lengths, samples_per_prompt, dtype_bytes)
        return weight_bytes + kv_write_bytes + kv_read_bytes

    def memory_bytes(
        self,
        prompt_lengths: list[int],
        num_engines: int,
        num_gpus_per_engine: int,
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        dtype_bytes: int = 2,
    ) -> int:
        """Approximate total HBM bytes moved per engine for prefill + decode.

        When multiple engines process work in parallel, this calculates the bytes
        moved by ONE engine processing its fraction of the prompts.

        Args:
            prompt_lengths: List of ALL prompt lengths across all engines
            num_engines: Number of vLLM engines working in parallel
            num_gpus_per_engine: Number of GPUs per engine (tensor parallelism)
            response_lengths: List of response lengths (samples_per_prompt * len(prompt_lengths) total)
            samples_per_prompt: Number of samples generated per prompt
            dtype_bytes: Bytes per element (2 for FP16/BF16)

        Returns:
            Memory bytes moved by ONE engine (not total across all engines)

        Assumptions:
          - Prompts are evenly distributed across engines
          - Each engine processes its subset independently
          - Weights are read once per token per layer (Q,K,V,O + MLP up/down)
          - KV cache: write K/V for every token; during decode, read all past K/V per new token
          - When batching samples, prompt KV cache is shared across samples
          - Embedding and LM head reads are ignored (usually dominated by matmul weight traffic)
        """
        if num_engines < 1:
            raise ValueError(f"num_engines must be >= 1, got {num_engines}")
        if num_gpus_per_engine < 1:
            raise ValueError(f"num_gpus_per_engine must be >= 1, got {num_gpus_per_engine}")

        if not prompt_lengths:
            return 0

        def _split_evenly(seq: list[int], parts: int) -> list[list[int]]:
            base, extra = divmod(len(seq), parts)
            result: list[list[int]] = []
            start = 0
            for i in range(parts):
                size = base + (1 if i < extra else 0)
                result.append(seq[start : start + size])
                start += size
            return result

        prompt_chunks = _split_evenly(prompt_lengths, num_engines)

        response_chunks: list[list[int] | None]
        if response_lengths is not None:
            assert len(response_lengths) == len(prompt_lengths) * samples_per_prompt, (
                f"Expected {len(prompt_lengths) * samples_per_prompt} response lengths, got {len(response_lengths)}"
            )
            response_chunks = []
            response_idx = 0
            for chunk in prompt_chunks:
                num_responses = len(chunk) * samples_per_prompt
                response_chunks.append(response_lengths[response_idx : response_idx + num_responses])
                response_idx += num_responses
        else:
            response_chunks = [None] * num_engines

        per_engine_totals: list[int] = []
        for chunk_prompts, chunk_responses in zip(prompt_chunks, response_chunks):
            if not chunk_prompts:
                per_engine_totals.append(0)
                continue

            total = self.prefill_memory_bytes(chunk_prompts, dtype_bytes)
            if chunk_responses is not None:
                total += self.decode_memory_bytes(chunk_prompts, chunk_responses, samples_per_prompt, dtype_bytes)
            per_engine_totals.append(total)

        if len(per_engine_totals) < num_engines:
            per_engine_totals.extend([0] * (num_engines - len(per_engine_totals)))

        avg_bytes_per_engine = math.ceil(sum(per_engine_totals) / num_engines)
        return avg_bytes_per_engine

    def calculate_mfu(
        self,
        prompt_lengths: list[int],
        generation_time: float,
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        num_gpus: int = 1,
    ) -> float:
        total_flops = self.flops(prompt_lengths, response_lengths, samples_per_prompt=samples_per_prompt)
        flops_per_second = total_flops / generation_time if generation_time > 0 else 0
        total_device_flops = self.device_flops * num_gpus
        return 100 * flops_per_second / total_device_flops

    def calculate_mbu(
        self,
        prompt_lengths: list[int],
        generation_time: float,
        response_lengths: list[int] | None = None,
        samples_per_prompt: int = 1,
        num_engines: int = 1,
        num_gpus_per_engine: int = 1,
    ) -> float:
        total_memory_bytes = self.memory_bytes(
            prompt_lengths,
            num_engines,
            num_gpus_per_engine,
            response_lengths=response_lengths,
            samples_per_prompt=samples_per_prompt,
        )
        bytes_per_second = total_memory_bytes / generation_time if generation_time > 0 else 0
        # Normalize against total system bandwidth. This is correct because prompt_lengths and
        # generation_time represent aggregated data from all engines already.
        total_device_bandwidth = self.device_memory_bandwidth * num_engines * num_gpus_per_engine
        return 100 * bytes_per_second / total_device_bandwidth

    def calculate_actor_utilization(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int],
        total_generation_time: float,
        samples_per_prompt: int,
        num_engines: int,
        num_gpus_per_engine: int,
    ) -> dict[str, float]:
        actor_mfu = self.calculate_mfu(
            prompt_lengths,
            total_generation_time,
            response_lengths=response_lengths,
            samples_per_prompt=samples_per_prompt,
            num_gpus=num_engines * num_gpus_per_engine,
        )
        actor_mbu = self.calculate_mbu(
            prompt_lengths,
            total_generation_time,
            response_lengths=response_lengths,
            samples_per_prompt=samples_per_prompt,
            num_engines=num_engines,
            num_gpus_per_engine=num_gpus_per_engine,
        )

        check_calculation(
            actor_mfu,
            "Actor MFU",
            self,
            total_generation_time,
            prompt_lengths,
            response_lengths,
            samples_per_prompt,
            num_engines,
            num_gpus_per_engine,
        )

        check_calculation(
            actor_mbu,
            "Actor MBU",
            self,
            total_generation_time,
            prompt_lengths,
            response_lengths,
            samples_per_prompt,
            num_engines,
            num_gpus_per_engine,
        )

        return {"mfu": actor_mfu, "mbu": actor_mbu}

    def calculate_learner_utilization(
        self,
        prompt_lengths: list[int],
        response_lengths: list[int],
        training_time: float,
        samples_per_prompt: int,
        num_training_gpus: int,
    ) -> dict[str, float]:
        total_sequence_lengths = [
            prompt_lengths[i // samples_per_prompt] + response_len for i, response_len in enumerate(response_lengths)
        ]

        training_flops = self.flops(
            prompt_lengths=total_sequence_lengths, response_lengths=None, samples_per_prompt=1, is_training=True
        )

        training_flops_per_second = training_flops / training_time
        total_training_device_flops = self.device_flops * num_training_gpus
        learner_mfu = 100 * training_flops_per_second / total_training_device_flops

        check_calculation(
            learner_mfu, "Learner MFU", self, training_time, total_sequence_lengths, None, 1, 1, num_training_gpus
        )

        return {"mfu": learner_mfu}

    def approximate_learner_utilization(
        self, total_tokens: int, avg_sequence_length: float, training_time: float, num_training_gpus: int
    ) -> dict[str, float]:
        num_sequences = int(total_tokens / avg_sequence_length)
        sequence_lengths = [int(avg_sequence_length)] * num_sequences

        training_flops = self.flops(
            prompt_lengths=sequence_lengths, response_lengths=None, samples_per_prompt=1, is_training=True
        )

        training_flops_per_second = training_flops / training_time
        total_training_device_flops = self.device_flops * num_training_gpus
        learner_mfu = 100 * training_flops_per_second / total_training_device_flops

        return {"mfu": learner_mfu}


def get_device_name(device_name: str) -> str:
    """Normalize a GPU device name to a standard key used in GPU_SPECS.

    The function converts device names from torch.cuda.get_device_name() format
    to a standardized key that can be used to look up GPU specifications.

    Args:
        device_name: Raw device name string (e.g., "NVIDIA H100 80GB HBM3")

    Returns:
        Standardized GPU key (e.g., "h100")

    Raises:
        ValueError: If the device name is not recognized

    Examples:
        >>> get_device_name("NVIDIA H100 80GB HBM3")
        'h100'

        >>> get_device_name("NVIDIA RTX PRO 6000 Blackwell Server Edition")
        'pro 6000'
    """
    normalized_device_name = device_name.lower().replace("-", " ")

    for key in GPU_SPECS:
        if key in normalized_device_name:
            return key
    raise ValueError(
        f"Unknown device name: {device_name}. Expected one of: {list(GPU_SPECS.keys())}. "
        f"Please raise an issue at https://github.com/allenai/open-instruct/issues with the device you need. In the interim, you can add the specs for your device using the name {normalized_device_name} to the GPU_SPECS dictionary in utils.py."
    )


def check_calculation(
    percentage: float,
    metric_name: str,
    model_dims: ModelDims,
    timing: float,
    prompt_lengths: list[int],
    response_lengths: list[int] | None,
    samples_per_prompt: int,
    num_engines: int,
    num_gpus_per_engine: int,
) -> None:
    if percentage <= 100:
        return

    import json

    full_device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths)
    avg_response_length = sum(response_lengths) / len(response_lengths) if response_lengths else 0

    test_case_json = {
        "model_name": "REPLACE_WITH_MODEL_NAME",
        "total_generation_time": timing,
        "samples_per_prompt": samples_per_prompt,
        "num_engines": num_engines,
        "num_gpus_per_engine": num_gpus_per_engine,
        "training_time": "REPLACE_WITH_TRAINING_TIME",
        "num_training_gpus": "REPLACE_WITH_NUM_TRAINING_GPUS",
        "prompt_lengths": prompt_lengths,
        "response_lengths": response_lengths,
    }

    warning_message = (
        f"{metric_name} exceeded 100%: {percentage:.2f}%\n"
        f"\n"
        f"{model_dims}\n"
        f"\n"
        f"Timing and GPU info:\n"
        f"  timing: {timing:.6f}s\n"
        f"  num_engines: {num_engines}\n"
        f"  num_gpus_per_engine: {num_gpus_per_engine}\n"
        f"  full_device_name: {full_device_name}\n"
        f"\n"
        f"Batch/sequence info:\n"
        f"  num_prompts: {len(prompt_lengths)}\n"
        f"  samples_per_prompt: {samples_per_prompt}\n"
        f"  avg_prompt_length: {avg_prompt_length:.1f}\n"
        f"  avg_response_length: {avg_response_length:.1f}\n"
        f"\n"
        f"To reproduce this calculation, use these exact parameters:\n"
        f"  prompt_lengths = {prompt_lengths}\n"
        f"  response_lengths = {response_lengths}\n"
        f"  timing = {timing}\n"
        f"  samples_per_prompt = {samples_per_prompt}\n"
        f"  num_engines = {num_engines}\n"
        f"  num_gpus_per_engine = {num_gpus_per_engine}\n"
        f"\n"
        f"JSON format for test case (copy this to mbu_reproduction_cases.json):\n"
        f"{json.dumps(test_case_json, indent=2)}\n"
        f"\n"
        f"This may indicate an issue with the MFU/MBU calculation logic or GPU specifications.\n"
        f"Please raise an issue at https://github.com/allenai/open-instruct/issues with the above information."
    )

    logger.warning(warning_message)


def combine_reward_metrics(reward_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Assumes same number of metric_records in each dict in the list"""
    buckets = defaultdict(list)
    for metrics in reward_metrics:
        for key, value in metrics.items():
            buckets[key].append(value)

    combined: dict[str, Any] = {}
    for key, records in buckets.items():
        sample_value = records[0]
        if isinstance(sample_value, np.ndarray):
            combined[key] = [x for value in records for x in value]
        elif isinstance(sample_value, (list | tuple)):
            concatenated: list[Any] = []
            for value in records:
                concatenated.extend(list(value))
            combined[key] = concatenated
        elif isinstance(sample_value, (int | float | bool | np.integer | np.floating)):
            # combine and get average value
            combined[key] = sum(value for value in records) / len(records) if len(records) > 0 else sample_value
        else:
            # Fallback: keep the latest value if aggregation strategy is unclear.
            combined[key] = records[-1]
    return combined


def send_slack_message(message: str) -> None:
    """Sends a message to a Slack webhook if configured.

    Args:
        message: Message body to send to Slack.
    """
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK")
    if not slack_webhook_url:
        logger.warning("SLACK_WEBHOOK environment variable not set. Skipping Slack alert.")
        return

    beaker_url = get_beaker_experiment_url()
    beaker_suffix = f" Check it out: {beaker_url}" if beaker_url else ""

    payload = {"text": f"{message}{beaker_suffix}"}
    try:
        response = requests.post(slack_webhook_url, json=payload)
        if not response.ok:
            logger.warning("Failed to send Slack alert with status %s: %s", response.status_code, response.text)
    except requests.RequestException as exc:
        logger.warning("Failed to send Slack alert due to network error: %s", exc)


def get_beaker_experiment_url() -> str | None:
    """If the env var BEAKER_WORKLOAD_ID is set, gets the current experiment URL."""
    try:
        beaker_client = beaker.Beaker.from_env()
        workload = beaker_client.workload.get(os.environ["BEAKER_WORKLOAD_ID"])
        url = beaker_client.experiment.url(workload.experiment)
        return url
    except Exception:
        return None


def get_denominator(loss_denominator: str | float) -> float | str:
    """
    Validates and converts the loss_denominator argument.
    """
    if loss_denominator == "token":
        return "token"

    val = float(loss_denominator)
    if val <= 0:
        raise ValueError(f"loss_denominator must be greater than 0 if not 'token', got: {loss_denominator}")
    return val
