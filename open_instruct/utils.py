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
import os
import sys
from dataclasses import dataclass, field
from typing import Any, List, NewType, Optional, Tuple

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk
from datasets.builder import DatasetGenerationError
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
def instruction_output_to_messages(example):
    """
    Convert an instruction in inst-output to a list of messages.
    e.g. vicgalle/alpaca-gpt4"""
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["messages"] = messages
    return example


def query_answer_to_messages(example):
    """
    Convert a query-answer pair to a list of messages.
    e.g. m-a-p/CodeFeedback-Filtered-Instruction"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["messages"] = messages
    return example


def query_response_to_messages(example):
    """
    Convert a query-response pair to a list of messages.
    e.g. meta-math/MetaMathQA"""
    messages = [
        {"role": "user", "content": example["query"]},
        {"role": "assistant", "content": example["response"]},
    ]
    example["messages"] = messages
    return example


def prompt_completion_to_messages(example):
    """
    Convert a prompt-completion pair to a list of messages.
    e.g. HuggingFaceH4/CodeAlpaca_20K"""
    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    example["messages"] = messages
    return example


def question_response_to_messages(example):
    """
    Convert a question-response pair to a list of messages.
    e.g. Open-Orca/OpenOrca"""
    messages = [
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
    name_mapping = {"gpt": "assistant", "user": "user", "human": "user"}
    messages = [{"role": name_mapping[conv["from"]], "content": conv["value"]} for conv in example["conversations"]]
    example["messages"] = messages
    return example


def get_datasets(
    dataset_mixer: dict,
    splits: Optional[List[str]] = None,
    configs: Optional[List[str]] = None,
    columns_to_keep: Optional[List[str]] = None,
    shuffle: bool = True,
    save_data_dir: Optional[str] = None,
    need_columns: Optional[List[str]] = None,
) -> DatasetDict:
    """
    Loads and mixes datasets according to proportions specified in `dataset_mixer`.

    Args:
        dataset_mixer (`dict`):
            Dictionary containing the dataset names and their training proportions.
            By default, all test proportions are 1.
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
    """
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
                    raise ValueError(f"Needed column {need_columns} not found in dataset {dataset.coulmn_names}.")

            # handle per-case conversions
            # if "instruction" and "output" columns are present and "messages" is not, convert to messages
            if (
                "instruction" in dataset.column_names
                and "output" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(instruction_output_to_messages, num_proc=10)
            elif (
                "prompt" in dataset.column_names
                and "completion" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(prompt_completion_to_messages, num_proc=10)
            elif "conversations" in dataset.column_names and "messages" not in dataset.column_names:
                dataset = dataset.map(conversations_to_messages, num_proc=10)
            elif (
                "question" in dataset.column_names
                and "response" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(question_response_to_messages, num_proc=10)
            elif (
                "query" in dataset.column_names
                and "answer" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(query_answer_to_messages, num_proc=10)
            elif (
                "query" in dataset.column_names
                and "response" in dataset.column_names
                and "messages" not in dataset.column_names
            ):
                dataset = dataset.map(query_response_to_messages, num_proc=10)

            # if id not in dataset, create it as ds-{index}
            if "id" not in dataset.column_names:
                id_col = [f"{ds}_{i}" for i in range(len(dataset))]
                dataset = dataset.add_column("id", id_col)

            # Remove redundant columns to avoid schema conflicts on load
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names if col not in (columns_to_keep + ["id"])]
            )

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

    # remove id column
    if len(raw_train_datasets) > 0:
        if "id" in raw_datasets["train"].column_names:
            raw_datasets["train"] = raw_datasets["train"].remove_columns("id")
    if len(raw_val_datasets) > 0:
        if "id" in raw_datasets["test"].column_names:
            raw_datasets["test"] = raw_datasets["test"].remove_columns("id")

    return raw_datasets


@dataclass
class FlatArguments:
    """
    Full arguments class for all fine-tuning jobs.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    dpo_use_paged_optimizer: bool = field(
        default=False,
        metadata={
            "help": "Use paged optimizer from bitsandbytes."
            " Not compatible with deepspeed (use deepspeed config instead)."
        },
    )
    dpo_beta: float = field(
        default=0.1,
        metadata={"help": "Beta parameter for DPO loss. Default is 0.1."},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    tokenizer_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention in the model training"},
    )
    use_slow_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the slow tokenizer or not (which is then fast tokenizer)."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                "This option should only be set to `True` for repositories you trust and in which you "
                "have read the code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, "
                "then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_mixer: Optional[dict] = field(
        default=None, metadata={"help": "A dictionary of datasets (local or HF) to sample from."}
    )
    dataset_mix_dir: Optional[str] = field(
        default=None, metadata={"help": "The directory to save the mixed dataset to disk."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a json/jsonl file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. "
                "Sequences longer than this will be truncated,"
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    add_bos: bool = field(
        default=False,
        metadata={
            "help": "Forcibly add bos token to the beginning of the input sequence."
            " Use only when tokenizer does not add bos token by default."
        },
    )
    clip_grad_norm: float = field(
        default=-1,
        metadata={"help": "Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead)."},
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(
        default=2e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer."},
    )
    logging_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Log the training loss and learning rate every logging_steps steps."},
    )
    lora_rank: int = field(
        default=64,
        metadata={"help": "The rank of lora."},
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "The alpha parameter of lora."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout rate of lora modules."},
    )
    lr_scheduler_type: str = field(
        default="linear",
        metadata={
            "help": "The scheduler type to use for learning rate adjustment.",
            "choices": ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        },
    )
    num_train_epochs: int = field(
        default=2,
        metadata={"help": "Total number of training epochs to perform."},
    )
    output_dir: str = field(
        default="output/",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    use_lora: bool = field(
        default=False,
        metadata={"help": "If True, will use LORA (low-rank parameter-efficient training) to train the model."},
    )
    use_qlora: bool = field(
        default=False,
        metadata={"help": "Use qLoRA training - initializes model in quantized form. Not compatible with deepspeed."},
    )
    use_8bit_optimizer: bool = field(
        default=False,
        metadata={"help": "Use 8bit optimizer from bitsandbytes. Not compatible with deepspeed."},
    )
    warmup_ratio: float = field(
        default=0.03,
        metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay for AdamW if we apply some."},
    )
    timeout: int = field(
        default=1800,
        metadata={
            "help": "Timeout for the training process in seconds."
            "Useful if tokenization process is long. Default is 1800 seconds (30 minutes)."
        },
    )
    reduce_loss: str = field(
        default="mean",
        metadata={
            "help": "How to reduce loss over tokens. Options are 'mean' or 'sum'."
            "Using 'sum' can improve chat model performance."
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "Entity to use for logging to wandb."},
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "If the training should continue from a checkpoint folder."},
    )
    with_tracking: bool = field(
        default=False,
        metadata={"help": "Whether to enable experiment trackers for logging."},
    )
    report_to: str = field(
        default="all",
        metadata={
            "help": "The integration to report results and logs to."
            "Options are 'tensorboard', 'wandb', 'comet_ml', 'clearml', or 'all'."
        },
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Turn on gradient checkpointing. Saves memory but slows training."},
    )
    max_train_steps: Optional[int] = field(
        default=None,
        metadata={"help": "If set, overrides the number of training steps. Otherwise, num_train_epochs is used."},
    )
    seed: int = field(default=42, metadata={"help": "Random seed for initialization and dataset shuffling."})
    checkpointing_steps: Optional[str] = field(
        default=None,
        metadata={
            "help": "Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch."  # noqa
        },
    )

    def __post_init__(self):
        if self.reduce_loss not in ["mean", "sum"]:
            raise ValueError("reduce_loss must be either 'mean' or 'sum'")
        if self.dataset_name is None and self.train_file is None and self.dataset_mixer is None:
            raise ValueError("Need either a dataset name, dataset mixer, or a training file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["json", "jsonl"], "`train_file` should be a json or a jsonl file."
        if (
            (self.dataset_name is not None and self.dataset_mixer is not None)
            or (self.dataset_name is not None and self.train_file is not None)
            or (self.dataset_mixer is not None and self.train_file is not None)
        ):
            raise ValueError("Cannot provide two dataset selection mechanisms.")


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

    def parse(self) -> DataClassType | Tuple[DataClassType]:
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
