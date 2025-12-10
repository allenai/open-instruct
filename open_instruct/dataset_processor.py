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
import multiprocessing
import os
from dataclasses import dataclass
from typing import Optional, Union

import torch
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

logging.basicConfig(level=logging.INFO)


# Preference dataset
INPUT_IDS_CHOSEN_KEY = "input_ids_chosen"
ATTENTION_MASK_CHOSEN_KEY = "attention_mask_chosen"
INPUT_IDS_REJECTED_KEY = "input_ids_rejected"
ATTENTION_MASK_REJECTED_KEY = "attention_mask_rejected"
INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
VERIFIER_SOURCE_KEY = "dataset"

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
    "deepseek_r1_zero": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer."
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think>"
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant:' }}"
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
    chat_template: Optional[str] = None

    # columns names for preference dataset
    preference_chosen_key: str = "chosen"
    preference_rejected_key: str = "rejected"

    # columns names for SFT dataset
    sft_messages_key: str = SFT_MESSAGE_KEY

    # columns name for the ground truth
    ground_truths_key: str = GROUND_TRUTHS_KEY

    # columns name for dataset source
    dataset_source_key: str = VERIFIER_SOURCE_KEY

    # filter config
    max_token_length: Optional[int] = None
    max_prompt_token_length: Optional[int] = None

    # dataset.map config
    sanity_check: bool = False
    sanity_check_max_samples: int = 100
    batched: bool = False
    load_from_cache_file: bool = True

    # Beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float (to parse the string) and then int to round down.
    num_proc: int = int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))

    # other config
    train_only_on_prompt: bool = False

    def __post_init__(self):
        if self.sanity_check:
            self.num_proc = 1
            self.load_from_cache_file = False

        if self.chat_template is not None and self.chat_template not in CHAT_TEMPLATES:
            raise ValueError(f"chat_template must None or one of {list(CHAT_TEMPLATES.keys())}")


def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus, dataset_len)


class DatasetProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizer, config: DatasetConfig) -> None:
        self.tokenizer = tokenizer
        self.config = config
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            logging.warning(
                "Tokenizer's pad token is the same as EOS token, this might cause the model to not learn to generate EOS tokens."
            )

    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        raise NotImplementedError

    def filter(self, dataset: DatasetDict):
        if self.config is None:
            logging.warning("No config provided, skipping filtering")
            return dataset
        raise NotImplementedError


class PreferenceDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Union[Dataset, DatasetDict]):
        def tokenize_fn(row):
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(
                row[self.config.preference_chosen_key][:-1], add_generation_prompt=True
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


class SFTDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):  # type: ignore[override]
        def tokenize_fn(row):
            if len(row[self.config.sft_messages_key]) == 1:
                prompt = row[self.config.sft_messages_key]
            else:
                prompt = row[self.config.sft_messages_key][:-1]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
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

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):  # type: ignore[override]
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


class SFTGroundTruthDatasetProcessor(DatasetProcessor):
    def tokenize(self, dataset: Dataset):  # type: ignore[override]
        def tokenize_fn(row):
            if len(row[self.config.sft_messages_key]) == 1:
                prompt = row[self.config.sft_messages_key]
            else:
                prompt = row[self.config.sft_messages_key][:-1]
            row[INPUT_IDS_PROMPT_KEY] = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            row[INPUT_IDS_KEY] = self.tokenizer.apply_chat_template(row[self.config.sft_messages_key])
            row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
            labels = copy.deepcopy(row[INPUT_IDS_KEY])
            if self.config.train_only_on_prompt:
                labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
            row[LABELS_KEY] = labels
            row[GROUND_TRUTHS_KEY] = row[self.config.ground_truths_key]
            row[VERIFIER_SOURCE_KEY] = row[self.config.dataset_source_key]
            return row

        return dataset.map(
            tokenize_fn,
            num_proc=get_num_proc(len(dataset), self.config.num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            load_from_cache_file=self.config.load_from_cache_file,
            desc="Tokenizing and reformatting SFT data",
        )

    def filter(self, dataset: Dataset, need_contain_labels: bool = True):  # type: ignore[override]
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
