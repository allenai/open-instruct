# this file deals with dataset pre-processing before training

# 1. PPO (prompt)
# 2. SFT (prompt + demonstration), there is also packing.
# 3. ✅ RM / DPO (chosen and rejected)
# 4. ✅ Visualization of length distributions?
# 5. ✅ Filter?
#   * Smart truncation?
# 6. ✅ dataset_num_proc
# 7. ✅ check EOS token
# 8. dataset mixer?
# 9. ✅ pretty print that show tokenization?
# 10. ✅ hashable tokneization?
# 11. inputs / labels / attention_mask
# 12. ✅ always set a `tokenizer.pad_token_id`?
# 13. a new DataCollatorForLanguageModeling?
# 14. `add_bos_token` and `add_eos_token`? E.g., LLAMA models
# 15. generate properties: has eos_token, bos_token?

# too many names related to "maximum length":
# * `max_seq_length` in SFT
# * `max_length`, `max_target_length` in RM / DPO,
# * `max_prompt_length` in DPO


import copy
import logging
import math
import multiprocessing
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import torch
from datasets import Dataset, DatasetDict
from rich.console import Console
from rich.text import Text
from tqdm import tqdm
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)


COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]
# Preference dataset
INPUT_IDS_CHOSEN_KEY = "input_ids_chosen"
ATTENTION_MASK_CHOSEN_KEY = "attention_mask_chosen"
INPUT_IDS_REJECTED_KEY = "input_ids_rejected"
ATTENTION_MASK_REJECTED_KEY = "attention_mask_rejected"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"

# NOTE (Costa): the `INPUT_IDS_PROMPT_KEY` is just for visualization purposes only
# also we don't really need `ATTENTION_MASK_CHOSEN_KEY` and `ATTENTION_MASK_REJECTED_KEY`
# since we are always padding from the right with a collator; however they might become
# more useful if we want to do some sort of packing in the future. The nice thing is
# that the tokenization logic would work for both DPO and RM training.
TOKENIZED_PREFERENCE_DATASET_KEYS = [
    INPUT_IDS_CHOSEN_KEY,
    INPUT_IDS_REJECTED_KEY,
    # ATTENTION_MASK_CHOSEN_KEY,
    # ATTENTION_MASK_REJECTED_KEY,
    # INPUT_IDS_PROMPT_KEY,
    # ATTENTION_MASK_PROMPT_KEY,
]


# SFT dataset
SFT_MESSAGE_KEY = "messages"
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"

# Binary dataset
BINARY_LABEL_KEY = "binary_labels"
BINARY_DATASET_KEYS = [
    INPUT_IDS_KEY,
    LABELS_KEY,
    BINARY_LABEL_KEY,
]

# Chat templates
# flake8: noqa
# note we added `{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}`
# because we want the template to not output eos_token if `add_generation_prompt=True`
CHAT_TEMPLATES = {
    "simple_concat_with_space": (
        "{% for message in messages %}"
        "{{ ' ' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_concat_with_new_line": (
        "{% for message in messages %}"
        "{{ '\n' if not loop.first else '' }}"
        "{{ message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "simple_chat": (
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] }}"
        "{% if loop.last and not add_generation_prompt %}{{ eos_token }}{% endif %}"
        "{% endfor %}"
    ),
    "assistant_message_only": (
        "{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "zephyr": (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + eos_token + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}
# flake8: noqa

# Performance tuning. Some rough numbers:
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130


@dataclass
class DatasetConfig:
    # dataset specs
    chat_template: str = "simple_chat"

    # columns names for preference dataset
    preference_chosen_key: str = "chosen"
    preference_rejected_key: str = "rejected"

    # columns names for SFT dataset
    sft_messages_key: str = SFT_MESSAGE_KEY

    # columns name for the ground truth
    ground_truths_key: str = GROUND_TRUTHS_KEY

    # columns name for dataset source
    dataset_source_key: str = DATASET_SOURCE_KEY

    # columns names for binary dataset
    binary_messages_key: str = SFT_MESSAGE_KEY
    label: str = BINARY_LABEL_KEY
    # extra setting for binary dataset
    convert_preference_to_binary_dataset: bool = False

    # filter config
    max_token_length: Optional[int] = None
    max_prompt_token_length: Optional[int] = None

    # dataset.map config
    sanity_check: bool = False
    sanity_check_max_samples: int = 100
    batched: bool = False
    load_from_cache_file: Optional[bool] = None
    num_proc: Optional[int] = None

    # other config
    train_only_on_prompt: bool = False

    # visualization configs
    ncols: int = 2

    def __post_init__(self):
        if self.sanity_check:
            self.num_proc = 1
            self.load_from_cache_file = False
        else:
            # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
            self.num_proc = int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))
            self.load_from_cache_file = True

        if self.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"chat_template must be one of {list(CHAT_TEMPLATES.keys())}")


def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)


def select_nested(dataset: DatasetDict, max_examples_per_split: int):
    """select the dataset nested in a DatasetDict"""
    return {key: dataset[key].select(range(min(max_examples_per_split, len(dataset[key])))) for key in dataset}


class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logging.warn(
                "Tokenizer's pad token is the same as EOS token, this might cause the model to not learn to generate EOS tokens."
            )

    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        raise NotImplementedError

    def filter(self, dataset: DatasetDict):
        if self.config is None:
            logging.warn("No config provided, skipping filtering")
            return dataset
        raise NotImplementedError

    def get_token_length_stats(self, features: list[str], dataset: Union[Dataset, DatasetDict]):
        """Get token length statistics for the dataset"""
        if isinstance(dataset, Dataset):
            return self._get_token_length_stats(features, dataset)
        elif isinstance(dataset, DatasetDict):
            stats = {}
            for key in dataset:
                stats[key] = self._get_token_length_stats(features, dataset[key])
            return stats

    def _get_token_length_stats(self, features: list[str], dataset: Dataset):
        stats = {}
        for key in features:
            stats[key] = {
                "max_token_length": max(len(x) for x in dataset[key]),
                "min_token_length": min(len(x) for x in dataset[key]),
                "mean_token_length": sum(len(x) for x in dataset[key]) / len(dataset[key]),
            }
        return stats

    def get_token_length_visualization(
        self,
        features: list[str],
        dataset: DatasetDict,
        save_path: str = "tmp.png",
        bins: int = 30,
    ):
        """Visualize the token length distribution of the dataset"""
        num_splits = len(dataset)
        cols = min(3, num_splits)  # Maximum 3 columns
        rows = math.ceil(num_splits / cols)

        fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
        fig.suptitle("Token Length Distribution", fontsize=16)

        for idx, (split_name, item) in enumerate(dataset.items()):
            row = idx // cols
            col = idx % cols
            ax = axs[row, col]

            for feature in features:
                token_lengths = [len(x) for x in item[feature]]
                ax.hist(
                    token_lengths,
                    bins=bins,
                    alpha=0.5,
                    label=feature,
                    edgecolor="black",
                )

            ax.set_title(f"{split_name} split")
            ax.set_xlabel("Token Length")
            ax.set_ylabel("Frequency")
            ax.legend(loc="upper right")

        plt.tight_layout()
        plt.savefig(save_path)
        logging.info(f"Saved token length distribution plot to {save_path}")
        plt.close(fig)  # Close the figure to free up memory


class PreferenceDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        def tokenize_fn(row):
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                row[self.config.preference_chosen_key][:-1],
                add_generation_prompt=True,
            )
            row[ATTENTION_MASK_PROMPT_KEY] = [1] * len(row[INPUT_IDS_PROMPT_KEY])
            row[INPUT_IDS_CHOSEN_KEY] = self.tokenizer.apply_chat_template(row[self.config.preference_chosen_key])
            row[ATTENTION_MASK_CHOSEN_KEY] = [1] * len(row[INPUT_IDS_CHOSEN_KEY])
            row[INPUT_IDS_REJECTED_KEY] = self.tokenizer.apply_chat_template(row[self.config.preference_rejected_key])
            row[ATTENTION_MASK_REJECTED_KEY] = [1] * len(row[INPUT_IDS_REJECTED_KEY])
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )

    def filter(self, dataset: Union[Dataset, DatasetDict]):
        def filter_fn(row):
            return (
                len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length
                if self.config.max_prompt_token_length is not None
                else (
                    True and len(row[INPUT_IDS_CHOSEN_KEY]) <= self.config.max_token_length
                    if self.config.max_token_length is not None
                    else (
                        True and len(row[INPUT_IDS_REJECTED_KEY]) <= self.config.max_token_length
                        if self.config.max_token_length is not None
                        else True
                    )
                )
            )

        filtered_dataset = dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
        )
        if isinstance(dataset, DatasetDict):
            for key in dataset:
                filtered_count = len(dataset[key]) - len(filtered_dataset[key])
                total_count = len(dataset[key])
                percentage = (filtered_count / total_count) * 100 if total_count > 0 else 0
                logging.info(f"Filtered out {filtered_count} samples or {percentage:.2f}% samples from {key}")
        return filtered_dataset

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(
            features=[
                INPUT_IDS_PROMPT_KEY,
                INPUT_IDS_CHOSEN_KEY,
                INPUT_IDS_REJECTED_KEY,
            ],
            dataset=dataset,
        )

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[
                INPUT_IDS_PROMPT_KEY,
                INPUT_IDS_CHOSEN_KEY,
                INPUT_IDS_REJECTED_KEY,
            ],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SFTDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            if len(row[self.config.sft_messages_key]) == 1:
                prompt = row[self.config.sft_messages_key]
            else:
                prompt = row[self.config.sft_messages_key][:-1]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
            row[LABELS_KEY] = labels
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


class SFTGroundTruthDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):
        def tokenize_fn(row):
            if len(row[self.config.sft_messages_key]) == 1:
                prompt = row[self.config.sft_messages_key]
            else:
                prompt = row[self.config.sft_messages_key][:-1]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
            )
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
            row[LABELS_KEY] = labels
            row[GROUND_TRUTHS_KEY] = row[self.config.ground_truths_key]
            row[DATASET_SOURCE_KEY] = row[self.config.dataset_source_key]
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):
        def filter_fn(row):
            max_prompt_token_length_ok = True
            if self.config.max_prompt_token_length is not None:
                max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= self.config.max_prompt_token_length

            max_token_length_ok = True
            if self.config.max_token_length is not None:
                max_token_length_ok = len(row[INPUT_IDS_KEY]) <= self.config.max_token_length

            contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
            return (
                max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)
            )

        return dataset.filter(
            filter_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Filtering SFT data",
        )

    def get_token_length_stats(self, dataset: Union[Dataset, DatasetDict]):
        return super().get_token_length_stats(features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY], dataset=dataset)

    def get_token_length_visualization(self, dataset: DatasetDict, save_path: str = "tmp.png", bins: int = 30):
        return super().get_token_length_visualization(
            features=[INPUT_IDS_PROMPT_KEY, INPUT_IDS_KEY],
            dataset=dataset,
            save_path=save_path,
            bins=bins,
        )


def convert_preference_dataset_to_binary_dataset(ds: Dataset):
    binary_ds = defaultdict(list)
    for i in tqdm(range(len(ds))):
        binary_ds[SFT_MESSAGE_KEY].append(ds[i]["chosen"])
        binary_ds[BINARY_LABEL_KEY].append(True)
        binary_ds[SFT_MESSAGE_KEY].append(ds[i]["rejected"])
        binary_ds[BINARY_LABEL_KEY].append(False)
    return Dataset.from_dict(binary_ds)


def visualize_token(tokens: list[int], tokenizer: PreTrainedTokenizer):
    i = 0
    console = Console()
    rich_text = Text()
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)
    console.print(rich_text)


class SimplePreferenceCollator:
    def __init__(self, pad_token_id: int):
        """Simple collator for preference dataset (always pad from the RIGHT)"""
        self.pad_token_id = pad_token_id

    def __call__(self, batch: List[Dict[str, int]]):
        """the input will have input_ids_chosen, input_ids_rejected"""
        # Find max length in the batch
        max_length_chosen = -1
        max_length_rejected = -1
        for i in range(len(batch)):
            max_length_chosen = max(max_length_chosen, len(batch[i]["input_ids_chosen"]))
            max_length_rejected = max(max_length_rejected, len(batch[i]["input_ids_rejected"]))
        max_length = max(max_length_chosen, max_length_rejected)
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences_chosen = []
        padded_sequences_rejected = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length_chosen = max_length - len(batch[i][INPUT_IDS_CHOSEN_KEY])
            pad_length_rejected = max_length - len(batch[i][INPUT_IDS_REJECTED_KEY])

            # Pad from the right
            padding_chosen = [self.pad_token_id] * pad_length_chosen
            padding_rejected = [self.pad_token_id] * pad_length_rejected
            padded_sequence_chosen = batch[i][INPUT_IDS_CHOSEN_KEY] + padding_chosen
            padded_sequence_rejected = batch[i][INPUT_IDS_REJECTED_KEY] + padding_rejected
            padded_sequences_chosen.append(padded_sequence_chosen)
            padded_sequences_rejected.append(padded_sequence_rejected)

        # Convert to tensors
        padded_sequences_chosen = torch.tensor(padded_sequences_chosen)
        padded_sequences_rejected = torch.tensor(padded_sequences_rejected)

        return {
            INPUT_IDS_CHOSEN_KEY: padded_sequences_chosen,
            INPUT_IDS_REJECTED_KEY: padded_sequences_rejected,
        }


class SimpleGenerateCollator:
    """Simple collator for generation task (always pad from the LEFT)"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]):
        """the input will have input_ids_prompt"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the left
            padding = [self.pad_token_id] * pad_length
            padded_sequence = padding + batch[i][INPUT_IDS_PROMPT_KEY]
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences = torch.tensor(padded_sequences)

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences,
        }


class SimpleGenerateCollatorWithGroundTruth:
    """Simple collator for generation task (always pad from the LEFT)"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict]):
        """the input will have input_ids_prompt"""
        # Find max length in the batch
        max_length = -1
        for i in range(len(batch)):
            max_length = max(max_length, len(batch[i][INPUT_IDS_PROMPT_KEY]))
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length = max_length - len(batch[i][INPUT_IDS_PROMPT_KEY])

            # Pad from the left
            padding = [self.pad_token_id] * pad_length
            padded_sequence = padding + batch[i][INPUT_IDS_PROMPT_KEY]
            padded_sequences.append(padded_sequence)

        # Convert to tensors
        padded_sequences = torch.tensor(padded_sequences)

        # ground truths
        ground_truths = [x[GROUND_TRUTHS_KEY] for x in batch]

        # datasets
        datasets = [x[DATASET_SOURCE_KEY] for x in batch]

        return {
            INPUT_IDS_PROMPT_KEY: padded_sequences,
            GROUND_TRUTHS_KEY: ground_truths,
            DATASET_SOURCE_KEY: datasets,
        }


if __name__ == "__main__":
    # too little data; it should just use 1 CPU despite the number of available CPUs
    assert get_num_proc(296, 120, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU) == 1

    # try to determine the number of CPUs to use
    assert get_num_proc(1500, 120, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU) == 3

    # too much data; it should use all available CPUs
    assert get_num_proc(1000000, 120, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU) == 120
