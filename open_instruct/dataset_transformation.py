# this file deals with dataset pre-processing before training

# 1. PPO (prompt)
# 2. SFT (prompt + demonstration), there is also packing.
# 3. ✅ RM / DPO (chosen and rejected)
# 4. ✅ Visualization of length distributions?
# 5. ✅ Filter?
# 6. ✅ dataset_num_proc
# 7. ✅ check EOS token
# 8. dataset mixer?
# 9. ✅ pretty print that show tokenization?
# 10. ✅ hashable tokneization?
# 11. inputs / labels / attention_mask
# 12. ✅ always set a `tokenizer.pad_token_id`?
# 13. a new DataCollatorForLanguageModeling?
# 14. ✅ `add_bos_token` and `add_eos_token`? E.g., LLAMA models
# 15. ✅ generate properties: has eos_token, bos_token (through chat template)

# ✅ get tokenizer revision
# ✅ get dataset revision
# create a cached tokenized dataset, with tokenized revision, dataset revision, tokenization function name.

# too many names related to "maximum length":
# * `max_seq_length` in SFT
# * `max_length`, `max_target_length` in RM / DPO,
# * `max_prompt_length` in DPO

# TODO: note that tokenizer doesn't change but model name does change. Should be mindful of this.
"""
This file contains the utility to transform and cache datasets with different configurations.
The main things we are looking for are:
* handle dataset mixing
* handle different tokenization functions
* **cache** the tokenized dataset so we don't have to re-tokenize every time
    * This is especially important when we have 405B SFT models: 32 nodes are just spending like
    5 minutes to tokenize the dataset. This translates to 32 * 5 * 8 = 1280 minutes = 21 hours of
    wasted H100 time.
    * Sometimes we also launch on places that don't have a shared cache (e.g., GCP), so we would
    download individual datasets 32 times, and wait for concatenation and tokenization (actually 
    twice because the `with accelerator.main_process_first()` function assumes a shared cache)
"""

import copy
import hashlib
import json
import multiprocessing
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi, ModelCard, revision_exists
from rich.console import Console
from rich.text import Text
from transformers import (
    AutoTokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
)
from transformers.utils.hub import cached_file, extract_commit_hash


# ----------------------------------------------------------------------------
# Utilities
def get_commit_hash(model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"):
    file = cached_file(model_name_or_path, revision=revision, filename=filename, repo_type=repo_type)
    commit_hash = extract_commit_hash(file, None)
    return commit_hash


# Performance tuning. Some rough numbers:
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130


def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus)


COLORS = ["on red", "on green", "on blue", "on yellow", "on magenta"]


def visualize_token(tokens: list[int], tokenizer: PreTrainedTokenizer):
    i = 0
    console = Console()
    rich_text = Text()
    for i, token in enumerate(tokens):
        color = COLORS[i % len(COLORS)]
        decoded_token = tokenizer.decode(token)
        rich_text.append(f"{decoded_token}", style=color)
    console.print(rich_text)


# ----------------------------------------------------------------------------
# Tokenization
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
    # template is taken from https://arxiv.org/abs/2501.12948.
    "r1_simple_chat": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
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
    "r1_simple_chat_postpend_think": (
        "A conversation between User and Assistant. "
        "The user asks a question, and the Assistant solves it. "
        "The assistant first thinks about the reasoning process in "
        "the mind and then provides the user with the answer. "
        "The reasoning process and answer are enclosed within <think> </think> "
        "and <answer> </answer> tags, respectively, "
        "i.e., <think> reasoning process here </think> "
        "<answer> answer here </answer>."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': ' + message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}
# flake8: noqa


def get_tokenizer_simple_v1(tc: "TokenizerConfig"):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.model_name_or_path,
        revision=tc.revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    return tokenizer


def get_tokenizer_tulu_v1(tc: "TokenizerConfig"):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.model_name_or_path,
        revision=tc.revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # only add if the pad token is not present already.
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )
        assert num_added_tokens in [
            0,
            1,
        ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert tc.add_bos, "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens(
                {
                    "pad_token": "<pad>",
                }
            )
            assert (
                num_added_tokens <= 1
            ), "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
    # NOTE: (Costa) I just commented the `OPTForCausalLM` because we are not likely to use it.
    # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
    #     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
    elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast) and tokenizer.pad_token is None:
        num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
        assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(tc.model_name_or_path).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.model_name_or_path}.")

    if tc.add_bos:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.bos_token is not None and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    return tokenizer


def get_tokenizer_tulu_v2_1(tc: "TokenizerConfig"):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.model_name_or_path,
        revision=tc.revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # only add if the pad token is not present already, or if the current one is set to eos_token_id.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens in [
                0,
                1,
            ], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            # OLMo newer models use this tokenizer
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
                assert (
                    tc.add_bos
                ), "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
            # else, pythia / other models
            else:
                num_added_tokens = tokenizer.add_special_tokens(
                    {
                        "pad_token": "<pad>",
                    }
                )
                assert (
                    num_added_tokens <= 1
                ), "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
        # NOTE: (Costa) I just commented the `OPTForCausalLM` because we are not likely to use it.
        # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        #     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
        elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    assert (
        tokenizer.pad_token_id != tokenizer.eos_token_id
    ), "pad token and eos token matching causes issues in our setup."

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(tc.model_name_or_path).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.model_name_or_path}.")

    if tc.add_bos:
        if tokenizer.chat_template.startswith("{{ bos_token }}") or (
            tokenizer.bos_token is not None and tokenizer.chat_template.startswith(tokenizer.bos_token)
        ):
            raise ValueError(
                "You specified add_bos=True, but the chat template already has a bos_token at the beginning."
            )
        # also add bos in the chat template if not already there
        tokenizer.chat_template = "{{ bos_token }}" + tokenizer.chat_template

    return tokenizer


GET_TOKENIZER_FN = {
    "get_tokenizer_simple_v1": get_tokenizer_simple_v1,
    "get_tokenizer_tulu_v1": get_tokenizer_tulu_v1,  # old version, see https://github.com/allenai/open-instruct/pull/570
    "get_tokenizer_tulu_v2_1": get_tokenizer_tulu_v2_1,
}


@dataclass
class TokenizerConfig:
    model_name_or_path: str
    revision: str
    trust_remote_code: bool = True
    use_fast: bool = True
    chat_template_name: Optional[str] = None  # TODO: should I give an option to force override?
    add_bos: bool = False
    get_tokenizer_fn: str = "get_tokenizer_tulu_v2_1"

    # for tracking purposes
    tokenizer_commit_hash: Optional[str] = None

    def __post_init__(self):
        self.tokenizer_commit_hash = get_commit_hash(
            self.model_name_or_path, self.revision, filename="tokenizer_config.json"
        )
        self.tokenizer = GET_TOKENIZER_FN[self.get_tokenizer_fn](self)


# TODO: for testing, we should load the tokenizer from the sft / dpo / rl and make sure they are all the same.


# ----------------------------------------------------------------------------
# Dataset Transformation
# SFT dataset
DEFAULT_SFT_MESSAGES_KEY = "messages"
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
TOKENIZED_SFT_DATASET_KEYS = [
    INPUT_IDS_KEY,
    ATTENTION_MASK_KEY,
    LABELS_KEY,
]

# Preference dataset
# NOTE (Costa): the `INPUT_IDS_PROMPT_KEY` is just for visualization purposes only
# also we don't really need `CHOSEN_ATTENTION_MASK_KEY` and `REJECTED_ATTENTION_MASK_KEY`
# since we are always padding from the right with a collator; however they might become
# more useful if we want to do some sort of packing in the future. The nice thing is
# that the tokenization logic would work for both DPO and RM training.
DEFAULT_CHOSEN_KEY = "chosen"
DEFAULT_REJECTED_KEY = "rejected"
CHOSEN_INPUT_IDS_KEY = "chosen_input_ids"
CHOSEN_ATTENTION_MASK_KEY = "chosen_attention_mask"
CHOSEN_LABELS_KEY = "chosen_labels"
REJECTED_INPUT_IDS_KEY = "rejected_input_ids"
REJECTED_ATTENTION_MASK_KEY = "rejected_attention_mask"
REJECTED_LABELS_KEY = "rejected_labels"

INPUT_IDS_PROMPT_KEY = "input_ids_prompt"
ATTENTION_MASK_PROMPT_KEY = "attention_mask_prompt"
GROUND_TRUTHS_KEY = "ground_truth"
DATASET_SOURCE_KEY = "dataset"

TOKENIZED_PREFERENCE_DATASET_KEYS = [
    CHOSEN_INPUT_IDS_KEY,
    CHOSEN_LABELS_KEY,
    CHOSEN_ATTENTION_MASK_KEY,
    REJECTED_INPUT_IDS_KEY,
    REJECTED_LABELS_KEY,
    REJECTED_ATTENTION_MASK_KEY,
]


# TODO: allow passing in sft_message key, so we can train on "chosen" of pref dataset.
def sft_tokenize_v1(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY
):
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    return row


def sft_tokenize_mask_out_prompt_v1(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY
):
    """mask out the prompt tokens by manipulating labels"""
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    labels[: len(row[INPUT_IDS_PROMPT_KEY])] = [-100] * len(row[INPUT_IDS_PROMPT_KEY])
    row[LABELS_KEY] = labels
    return row


def sft_filter_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: Optional[int] = None,
    max_token_length: Optional[int] = None,
    need_contain_labels: bool = True,
):
    max_prompt_token_length_ok = True
    if max_prompt_token_length is not None:
        max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length

    max_token_length_ok = True
    if max_token_length is not None:
        max_token_length_ok = len(row[INPUT_IDS_KEY]) <= max_token_length

    contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
    return max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)


def sft_tulu_tokenize_and_truncate_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_seq_length: int):
    """taken directly from https://github.com/allenai/open-instruct/blob/ba11286e5b9eb00d4ce5b40ef4cac1389888416a/open_instruct/finetune.py#L385"""
    messages = row["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    row[INPUT_IDS_KEY] = input_ids.flatten()
    row[LABELS_KEY] = labels.flatten()
    row[ATTENTION_MASK_KEY] = attention_mask.flatten()
    return row


def sft_tulu_filter_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    return any(x != -100 for x in row[LABELS_KEY])


def preference_tokenize_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    # Extract prompt (all messages except the last one)
    prompt = row["chosen"][:-1]

    # Tokenize prompt
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[ATTENTION_MASK_PROMPT_KEY] = [1] * len(row[INPUT_IDS_PROMPT_KEY])

    # Tokenize chosen completion
    row[CHOSEN_INPUT_IDS_KEY] = tokenizer.apply_chat_template(row["chosen"])
    row[CHOSEN_ATTENTION_MASK_KEY] = [1] * len(row[CHOSEN_INPUT_IDS_KEY])

    # Tokenize rejected completion
    row[REJECTED_INPUT_IDS_KEY] = tokenizer.apply_chat_template(row["rejected"])
    row[REJECTED_ATTENTION_MASK_KEY] = [1] * len(row[REJECTED_INPUT_IDS_KEY])

    return row


def preference_filter_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_prompt_token_length: Optional[int] = None,
    max_token_length: Optional[int] = None,
):
    # Check prompt length if specified
    if max_prompt_token_length is not None:
        if len(row[INPUT_IDS_PROMPT_KEY]) > max_prompt_token_length:
            return False

    # Check total sequence lengths if specified
    if max_token_length is not None:
        if len(row[CHOSEN_INPUT_IDS_KEY]) > max_token_length:
            return False
        if len(row[REJECTED_INPUT_IDS_KEY]) > max_token_length:
            return False

    return True


def preference_tulu_tokenize_and_truncate_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_seq_length: int,
    chosen_key: str = DEFAULT_CHOSEN_KEY,
    rejected_key: str = DEFAULT_REJECTED_KEY,
):
    """
    Here we assume each example has a rejected and chosen field, both of which are a list of messages.
    Each message is a dict with 'role' and 'content' fields.
    We assume only the last message is different, and the prompt is contained in the list of messages.
    """
    chosen_messages = row[chosen_key]
    rejected_messages = row[rejected_key]
    if len(chosen_messages) == 0:
        raise ValueError("chosen messages field is empty.")
    if len(rejected_messages) == 0:
        raise ValueError("rejected messages field is empty.")

    chosen_encoded = sft_tulu_tokenize_and_truncate_v1(
        {DEFAULT_SFT_MESSAGES_KEY: chosen_messages}, tokenizer, max_seq_length
    )
    rejected_encoded = sft_tulu_tokenize_and_truncate_v1(
        {DEFAULT_SFT_MESSAGES_KEY: rejected_messages}, tokenizer, max_seq_length
    )

    return {
        CHOSEN_INPUT_IDS_KEY: chosen_encoded["input_ids"],
        CHOSEN_LABELS_KEY: chosen_encoded["labels"],
        CHOSEN_ATTENTION_MASK_KEY: chosen_encoded["attention_mask"],
        REJECTED_INPUT_IDS_KEY: rejected_encoded["input_ids"],
        REJECTED_LABELS_KEY: rejected_encoded["labels"],
        REJECTED_ATTENTION_MASK_KEY: rejected_encoded["attention_mask"],
    }


def preference_tulu_filter_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    return any(x != -100 for x in row[CHOSEN_LABELS_KEY]) and any(x != -100 for x in row[REJECTED_LABELS_KEY])


def rlvr_tokenize_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
    ground_truths_key: str = GROUND_TRUTHS_KEY,
    dataset_source_key: str = DATASET_SOURCE_KEY,
):
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
    )
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    row[GROUND_TRUTHS_KEY] = row[ground_truths_key]
    row[DATASET_SOURCE_KEY] = row[dataset_source_key]
    return row


def rlvr_filter_v1(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    need_contain_labels: bool = True,
    max_prompt_token_length: Optional[int] = None,
    max_token_length: Optional[int] = None,
):
    max_prompt_token_length_ok = True
    if max_prompt_token_length is not None:
        max_prompt_token_length_ok = len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length

    max_token_length_ok = True
    if max_token_length is not None:
        max_token_length_ok = len(row[INPUT_IDS_KEY]) <= max_token_length

    contain_some_labels = any(x != -100 for x in row[LABELS_KEY])
    return max_prompt_token_length_ok and max_token_length_ok and (contain_some_labels or not need_contain_labels)


TRANSFORM_FNS = {
    "sft_tokenize_v1": (sft_tokenize_v1, "map"),
    "sft_tokenize_mask_out_prompt_v1": (sft_tokenize_mask_out_prompt_v1, "map"),
    "sft_filter_v1": (sft_filter_v1, "filter"),
    "sft_tulu_tokenize_and_truncate_v1": (sft_tulu_tokenize_and_truncate_v1, "map"),
    "sft_tulu_filter_v1": (sft_tulu_filter_v1, "filter"),
    "preference_tokenize_v1": (preference_tokenize_v1, "map"),
    "preference_filter_v1": (preference_filter_v1, "filter"),
    "preference_tulu_tokenize_and_truncate_v1": (preference_tulu_tokenize_and_truncate_v1, "map"),
    "preference_tulu_filter_v1": (preference_tulu_filter_v1, "filter"),
    "rlvr_tokenize_v1": (rlvr_tokenize_v1, "map"),
    "rlvr_filter_v1": (rlvr_filter_v1, "filter"),
}


# ----------------------------------------------------------------------------
# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    dataset_range: Optional[int] = None
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # for tracking purposes
    dataset_commit_hash: Optional[str] = None

    def __post_init__(self):
        # if the file exists locally, use the local file
        if os.path.exists(self.dataset_name) and self.dataset_name.endswith(".jsonl"):
            assert self.dataset_split == "train", "Only train split is supported for local jsonl files."
            self.dataset = load_dataset(
                "json",
                data_files=self.dataset_name,
                split=self.dataset_split,
            )
        else:
            # commit hash only works for hf datasets
            self.dataset_commit_hash = get_commit_hash(
                self.dataset_name, self.dataset_revision, "README.md", "dataset"
            )
            self.dataset = load_dataset(
                self.dataset_name,
                split=self.dataset_split,
                revision=self.dataset_revision,
            )
        if self.dataset_range is None:
            dataset_range = len(self.dataset)
            self.update_range(dataset_range)

    def update_range(self, dataset_range: int):
        self.dataset_range = dataset_range
        if self.dataset_range > len(self.dataset):
            raise ValueError("Dataset range exceeds dataset length")
        self.dataset = self.dataset.select(range(self.dataset_range))


def get_dataset_v1(dc: DatasetConfig, tc: TokenizerConfig):
    # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
    num_proc = int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))

    tokenizer = tc.tokenizer
    dataset = dc.dataset

    for fn_name in dc.transform_fn:
        fn, fn_type = TRANSFORM_FNS[fn_name]
        # always pass in tokenizer and other args if needed
        fn_kwargs = {"tokenizer": tokenizer}
        target_columns = dataset.column_names
        if fn_name in dc.transform_fn_args:
            target_columns = dc.transform_fn_args[fn_name].pop("target_columns", dataset.column_names)
            fn_kwargs.update(dc.transform_fn_args[fn_name])

        # perform the transformation
        if fn_type == "map":
            dataset = dataset.map(
                fn,
                fn_kwargs=fn_kwargs,
                remove_columns=[col for col in dataset.column_names if col not in target_columns],
                num_proc=get_num_proc(len(dataset), num_proc, APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU),
            )
        elif fn_type == "filter":
            dataset = dataset.filter(
                fn,
                fn_kwargs=fn_kwargs,
                num_proc=get_num_proc(len(dataset), num_proc, FILTER_EXAMPLE_PER_SECOND_PER_CPU),
            )
        # NOTE: elif we can implement packing here to create a packed SFT dataset. Low priority for now.
        else:
            raise ValueError(f"Unknown transform function type: {fn_type}")

    if len(dataset) == 0:
        raise ValueError("No examples left after transformation")
    return dataset


class DatasetTransformationCache:
    def __init__(self, hf_entity: Optional[str] = None):
        self.hf_entity = hf_entity or HfApi().whoami()["name"]

    def compute_config_hash(self, dcs: List[DatasetConfig], tc: TokenizerConfig) -> str:
        """Compute a deterministic hash of both configs for caching."""
        dc_dicts = [{k: v for k, v in asdict(dc).items() if v is not None} for dc in dcs]
        tc_dict = {k: v for k, v in asdict(tc).items() if v is not None}
        combined_dict = {"dataset_configs": dc_dicts, "tokenizer_config": tc_dict}
        config_str = json.dumps(combined_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:10]

    def load_or_transform_dataset(self, dcs: List[DatasetConfig], tc: TokenizerConfig) -> Dataset:
        """Load dataset from cache if it exists, otherwise transform and cache it."""
        config_hash = self.compute_config_hash(dcs, tc)
        repo_name = f"{self.hf_entity}/dataset-mix-cached"

        # NOTE: the cached dataset is always train split
        DEFAULT_SPLIT_FOR_CACHED_DATASET = "train"

        # Check if the revision exists
        if revision_exists(repo_name, config_hash, repo_type="dataset"):
            print(f"✅ Found cached dataset at https://huggingface.co/datasets/{repo_name}/tree/{config_hash}")
            # Use the split from the first dataset config as default
            return load_dataset(repo_name, split=DEFAULT_SPLIT_FOR_CACHED_DATASET, revision=config_hash)

        print(f"Cache not found, transforming datasets...")

        # Transform each dataset
        transformed_datasets = []
        for dc in dcs:
            dataset = get_dataset_v1(dc, tc)
            transformed_datasets.append(dataset)

        # Combine datasets
        combined_dataset = concatenate_datasets(transformed_datasets)

        # Push to hub with config hash as revision
        combined_dataset.push_to_hub(
            repo_name,
            private=True,
            revision=config_hash,
            commit_message=f"Cache combined dataset with configs hash: {config_hash}",
        )
        print(f"🚀 Pushed transformed dataset to https://huggingface.co/datasets/{repo_name}/tree/{config_hash}")

        model_card = ModelCard(
            f"""\
---
tags: [open-instruct]
---

# Cached Tokenized Datasets

## Summary

This is a cached dataset produced by https://github.com/allenai/open-instruct

## Configuration

`TokenizerConfig`:
```json
{json.dumps(asdict(tc), indent=2)}
```

`List[DatasetConfig]`:
```json
{json.dumps([asdict(dc) for dc in dcs], indent=2)}
```
"""
        )
        model_card.push_to_hub(repo_name, repo_type="dataset", revision=config_hash)

        # NOTE: Load the dataset again to make sure it's downloaded to the HF cache
        print(f"✅ Found cached dataset at https://huggingface.co/datasets/{repo_name}/tree/{config_hash}")
        return load_dataset(repo_name, split=DEFAULT_SPLIT_FOR_CACHED_DATASET, revision=config_hash)


def get_cached_dataset(dcs: List[DatasetConfig], tc: TokenizerConfig, hf_entity: Optional[str] = None) -> Dataset:
    cache = DatasetTransformationCache(hf_entity=hf_entity)
    return cache.load_or_transform_dataset(dcs, tc)


def get_cached_dataset_tulu_sft(
    dataset_mixer_list: List[str],
    tc: TokenizerConfig,
    max_seq_length: int,
    hf_entity: Optional[str] = None,
) -> Dataset:
    dcs = []
    assert len(dataset_mixer_list) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer_list}"
    for i in range(0, len(dataset_mixer_list), 2):
        dataset_name = dataset_mixer_list[i]
        frac_or_num_samples = dataset_mixer_list[i + 1]
        if "." in frac_or_num_samples:
            frac_or_num_samples = float(frac_or_num_samples)
        else:
            frac_or_num_samples = int(frac_or_num_samples)

        dataset_config = DatasetConfig(
            dataset_name=dataset_name,
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tulu_tokenize_and_truncate_v1", "sft_tulu_filter_v1"],
            transform_fn_args={
                "sft_tulu_tokenize_and_truncate_v1": {
                    "max_seq_length": max_seq_length,
                    "target_columns": TOKENIZED_SFT_DATASET_KEYS,
                }
            },
        )
        if frac_or_num_samples > 1.0:
            new_range = int(frac_or_num_samples)
        else:
            new_range = int(frac_or_num_samples * len(dataset_config.dataset))
        dataset_config.update_range(new_range)
        dcs.append(dataset_config)
    cache = DatasetTransformationCache(hf_entity=hf_entity)
    return cache.load_or_transform_dataset(dcs, tc)


def get_cached_dataset_tulu_preference(
    dataset_mixer_list: List[str], tc: TokenizerConfig, max_seq_length: int, hf_entity: Optional[str] = None
) -> Dataset:
    dcs = []
    assert len(dataset_mixer_list) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer_list}"
    for i in range(0, len(dataset_mixer_list), 2):
        dataset_name = dataset_mixer_list[i]
        frac_or_num_samples = dataset_mixer_list[i + 1]
        if "." in frac_or_num_samples:
            frac_or_num_samples = float(frac_or_num_samples)
        else:
            frac_or_num_samples = int(frac_or_num_samples)

        dataset_config = DatasetConfig(
            dataset_name=dataset_name,
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["preference_tulu_tokenize_and_truncate_v1", "preference_tulu_filter_v1"],
            transform_fn_args={
                "preference_tulu_tokenize_and_truncate_v1": {
                    "max_seq_length": max_seq_length,
                    "target_columns": TOKENIZED_PREFERENCE_DATASET_KEYS,
                }
            },
        )
        if frac_or_num_samples > 1.0:
            new_range = int(frac_or_num_samples)
        else:
            new_range = int(frac_or_num_samples * len(dataset_config.dataset))
        dataset_config.update_range(new_range)
        dcs.append(dataset_config)
    cache = DatasetTransformationCache(hf_entity=hf_entity)
    return cache.load_or_transform_dataset(dcs, tc)


def get_cached_dataset_rlvr(
    dataset_mixer_list: List[str],
    dataset_mixer_list_splits: List[str],
    tc: TokenizerConfig,
    max_token_length: Optional[int] = None,
    max_prompt_token_length: Optional[int] = None,
    hf_entity: Optional[str] = None,
) -> Dataset:
    if len(dataset_mixer_list_splits) == 1:
        print("by default, we will use the same split for all datasets")
        dataset_mixer_list_splits = [dataset_mixer_list_splits[0]] * len(dataset_mixer_list)
    dcs = []
    assert len(dataset_mixer_list) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer_list}"
    for i in range(0, len(dataset_mixer_list), 2):
        dataset_name = dataset_mixer_list[i]
        frac_or_num_samples = dataset_mixer_list[i + 1]
        if "." in frac_or_num_samples:
            frac_or_num_samples = float(frac_or_num_samples)
        else:
            frac_or_num_samples = int(frac_or_num_samples)

        dataset_config = DatasetConfig(
            dataset_name=dataset_name,
            dataset_split=dataset_mixer_list_splits[i],
            dataset_revision="main",
            transform_fn=["rlvr_tokenize_v1", "rlvr_filter_v1"],
            transform_fn_args={
                "rlvr_filter_v1": {
                    "max_token_length": max_token_length,
                    "max_prompt_token_length": max_prompt_token_length,
                }
            },
        )
        if frac_or_num_samples > 1.0:
            new_range = int(frac_or_num_samples)
        else:
            new_range = int(frac_or_num_samples * len(dataset_config.dataset))
        dataset_config.update_range(new_range)
        dcs.append(dataset_config)
    cache = DatasetTransformationCache(hf_entity=hf_entity)
    return cache.load_or_transform_dataset(dcs, tc)


def test_sft_dpo_same_tokenizer():
    base_to_sft_tc = TokenizerConfig(
        model_name_or_path="meta-llama/Llama-3.1-8B", revision="main", chat_template_name="tulu"
    )
    sft_to_dpo_tc = TokenizerConfig(
        model_name_or_path="allenai/Llama-3.1-Tulu-3-8B-SFT", revision="main", chat_template_name="tulu"
    )
    dpo_to_rl_tc = TokenizerConfig(
        model_name_or_path="allenai/Llama-3.1-Tulu-3-8B-DPO", revision="main", chat_template_name="tulu"
    )

    def equal_tokenizer(tc1, tc2):
        tok1 = tc1.tokenizer
        tok2 = tc2.tokenizer
        assert tok1.vocab_size == tok2.vocab_size, "Vocab size should be the same"
        assert tok1.model_max_length == tok2.model_max_length, "Model max length should be the same"
        assert tok1.is_fast == tok2.is_fast, "is_fast should be the same"
        assert tok1.padding_side == tok2.padding_side, "padding_side should be the same"
        assert tok1.truncation_side == tok2.truncation_side, "truncation_side should be the same"
        assert (
            tok1.clean_up_tokenization_spaces == tok2.clean_up_tokenization_spaces
        ), "clean_up_tokenization_spaces should be the same"
        assert tok1.added_tokens_decoder == tok2.added_tokens_decoder, "added_tokens_decoder should be the same"

    equal_tokenizer(base_to_sft_tc, sft_to_dpo_tc)
    equal_tokenizer(sft_to_dpo_tc, dpo_to_rl_tc)
    equal_tokenizer(base_to_sft_tc, dpo_to_rl_tc)


def test_config_hash_different():
    """Test that different configurations produce different hashes."""
    tc = TokenizerConfig(model_name_or_path="meta-llama/Llama-3.1-8B", revision="main", chat_template_name="tulu")

    dcs1 = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1"],
            transform_fn_args={},
        )
    ]

    dcs2 = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_mask_out_prompt_v1"],
            transform_fn_args={},
        )
    ]

    cache = DatasetTransformationCache()
    hash1 = cache.compute_config_hash(dcs1, tc)
    hash2 = cache.compute_config_hash(dcs2, tc)
    assert hash1 != hash2, "Different configs should have different hashes"


def test_sft_dataset_caching():
    """Test caching functionality for SFT datasets."""
    tc = TokenizerConfig(model_name_or_path="meta-llama/Llama-3.1-8B", revision="main", chat_template_name="tulu")

    dcs = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1"],
            transform_fn_args={},
        ),
        DatasetConfig(
            dataset_name="allenai/tulu-3-hard-coded-10x",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1"],
            transform_fn_args={},
        ),
    ]

    # First transformation should cache
    dataset1 = get_cached_dataset(dcs, tc)

    # Second load should use cache
    dataset1_cached = get_cached_dataset(dcs, tc)

    # Verify the datasets are the same
    assert len(dataset1) == len(dataset1_cached), "Cached dataset should have same length"


def test_sft_different_transform():
    """Test different transform functions produce different cached datasets."""
    tc = TokenizerConfig(model_name_or_path="meta-llama/Llama-3.1-8B", revision="main", chat_template_name="tulu")

    dcs = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_mask_out_prompt_v1"],
            transform_fn_args={},
        ),
        DatasetConfig(
            dataset_name="allenai/tulu-3-hard-coded-10x",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_mask_out_prompt_v1"],
            transform_fn_args={},
        ),
    ]

    dataset = get_cached_dataset(dcs, tc)
    assert dataset is not None, "Should successfully create dataset with different transform"


def test_sft_filter():
    """Test different transform functions produce different cached datasets."""
    tc = TokenizerConfig(model_name_or_path="meta-llama/Llama-3.1-8B", revision="main", chat_template_name="tulu")

    ARBITRARY_MAX_LENGTH = 1000
    dcs = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-sft-personas-algebra",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["sft_tokenize_v1", "sft_filter_v1"],  # First tokenize, then filter
            transform_fn_args={
                "sft_filter_v1": {
                    "max_token_length": ARBITRARY_MAX_LENGTH  # Filter to sequences <= ARBITRARY_MAX_LENGTH tokens
                }
            },
        )
    ]

    filtered_dataset = get_cached_dataset(dcs, tc)
    # Verify that all sequences are <= ARBITRARY_MAX_LENGTH tokens
    max_length = max(len(example[INPUT_IDS_KEY]) for example in filtered_dataset)
    assert max_length <= ARBITRARY_MAX_LENGTH, f"Found sequence with length {max_length} > {ARBITRARY_MAX_LENGTH}"

    print("Filter test passed! Max sequence length:", max_length)
    print("All tests passed!")
    assert filtered_dataset is not None, "Should successfully create dataset with different transform"


def test_preference_dataset():
    """Test caching functionality for preference datasets."""
    tc = TokenizerConfig(model_name_or_path="meta-llama/Llama-3.1-8B", revision="main", chat_template_name="tulu")

    dcs_pref = [
        DatasetConfig(
            dataset_name="allenai/tulu-3-pref-personas-instruction-following",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["preference_tokenize_v1"],
            transform_fn_args={},
        ),
        DatasetConfig(
            dataset_name="allenai/tulu-3-wildchat-reused-on-policy-70b",
            dataset_split="train",
            dataset_revision="main",
            transform_fn=["preference_tokenize_v1"],
            transform_fn_args={},
        ),
    ]

    dataset_pref = get_cached_dataset(dcs_pref, tc)
    assert dataset_pref is not None, "Should successfully create preference dataset"


if __name__ == "__main__":
    test_sft_dpo_same_tokenizer()
    test_config_hash_different()
    test_sft_dataset_caching()
    test_sft_different_transform()
    test_preference_dataset()
    test_sft_filter()
    print("All tests passed!")
