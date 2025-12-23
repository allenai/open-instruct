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
from functools import cached_property
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import ModelCard, revision_exists
from rich.console import Console
from rich.text import Text
from transformers import (
    AutoConfig,
    AutoTokenizer,
    GPTNeoXTokenizerFast,
    LlamaTokenizer,
    LlamaTokenizerFast,
    PreTrainedTokenizer,
)
from transformers.utils.hub import _CACHED_NO_EXIST, TRANSFORMERS_CACHE, extract_commit_hash, try_to_load_from_cache

from open_instruct.utils import hf_whoami, max_num_processes


# ----------------------------------------------------------------------------
# Utilities
def custom_cached_file(model_name_or_path: str, filename: str, revision: str = None, repo_type: str = "model"):
    """@vwxyzjn: HF's `cached_file` no longer works for `repo_type="dataset"`."""
    # local_file = os.path.join(model_name_or_path, filename)

    if os.path.isdir(model_name_or_path):
        resolved_file = os.path.join(model_name_or_path, filename)
        if os.path.isfile(resolved_file):
            return resolved_file
        else:
            return None
    else:
        resolved_file = try_to_load_from_cache(
            model_name_or_path, filename, cache_dir=TRANSFORMERS_CACHE, revision=revision, repo_type=repo_type
        )
        # special return value from try_to_load_from_cache
        if resolved_file == _CACHED_NO_EXIST:
            return None
        return resolved_file


def get_commit_hash(
    model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"
) -> str:
    file = custom_cached_file(model_name_or_path, filename, revision=revision, repo_type=repo_type)
    commit_hash = extract_commit_hash(file, None)
    return commit_hash


def get_file_hash(
    model_name_or_path: str, revision: str, filename: str = "config.json", repo_type: str = "model"
) -> str:
    file = custom_cached_file(model_name_or_path, filename, revision=revision, repo_type=repo_type)
    if isinstance(file, str):
        with open(file, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    elif file is _CACHED_NO_EXIST:
        return f"{filename} not found"
    elif file is None:
        return f"{filename} not found"
    else:
        raise ValueError(f"Unexpected file type: {type(file)}")


def get_files_hash_if_exists(
    model_name_or_path: str, revision: str, filenames: List[str], repo_type: str = "model"
) -> List[str]:
    return [get_file_hash(model_name_or_path, revision, filename, repo_type) for filename in filenames]


# Performance tuning. Some rough numbers:
APPLY_CHAT_TEMPLATE_EXAMPLE_PER_SECOND_PER_CPU = 400
FILTER_EXAMPLE_PER_SECOND_PER_CPU = 1130


def get_num_proc(dataset_len: int, num_available_cpus: int, example_per_second_per_cpu) -> int:
    num_required_cpus = max(1, dataset_len // example_per_second_per_cpu)
    return min(num_required_cpus, num_available_cpus, dataset_len)


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


def visualize_token_role(tokens: list[int], masks: list[int], tokenizer: PreTrainedTokenizer):
    i = 0
    console = Console()
    rich_text = Text()
    # for i, token in enumerate():
    for i in range(min(len(tokens), len(masks))):
        token = tokens[i]
        color = COLORS[masks[i] % len(COLORS)]
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
    # olmo-core-compatible chat templates:
    # TODO: unify these 3 chat templates and send variables through the tokenizer's apply_chat_template kwargs
    "olmo": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
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
    "tulu_thinker": (
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% set content = message['content'] %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + content + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + content + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "tulu_thinker_r1_style": (
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
        "{% if message['role'] == 'system' %}"
        "{{ '<|system|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'user' %}"
        "{{ '<|user|>\n' + message['content'] + '\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{% set content = message['content'] %}"
        "{% if '</think>' in content %}"
        "{% set content = content.split('</think>')[-1] %}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|assistant|>\n' + content + eos_token + '\n' }}"
        "{% else %}"
        "{{ '<|assistant|>\n' + content + eos_token }}"
        "{% endif %}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|assistant|>\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    # olmo-core-compatible chat templates:
    # TODO: unify these 3 chat templates and send variables through the tokenizer's apply_chat_template kwargs
    "olmo_old": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are OLMo, a helpful function-calling AI assistant built by Ai2. Your date cutoff is November 2024, and your model weights are available at https://huggingface.co/allenai. You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_no_think_7b": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are Olmo, a helpful AI assistant built by Ai2. Your date cutoff is December 2024, and your model weights are available at https://huggingface.co/allenai.<|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_remove_intermediate_thinking": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% set content = message.get('content', none) %}"
        "{% if content is not none %}"
        "{% set content = content | string %}"
        "{% if not loop.last and '</think>' in content and '<think>' in content %}"
        "{% set content = content.split('</think>')[-1].lstrip('\\n') %}"
        "{% endif %}"
        "{{ content }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n<think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_no_think_sft_tokenization": (
        "{% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}"
        "{% if not has_system %}"
        "{{ '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}"
        "{% endif %}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ '<|im_start|>system\n' + message['content'] }}"
        "{% if message.get('functions', none) is not none %}"
        "{{ ' <functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ ' You do not currently have access to any functions. <functions></functions><|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'user' %}"
        "{% if message.get('functions', none) is not none %}"
        "{{ '<|im_start|>user\n' + message['content'] + '\n' + '<functions>' + message['functions'] + '</functions><|im_end|>\n' }}"
        "{% else %}"
        "{{ '<|im_start|>user\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% if message.get('content', none) is not none %}"
        "{{ message['content'] }}"
        "{% endif %}"
        "{% if message.get('function_calls', none) is not none %}"
        "{{ '<function_calls>' + message['function_calls'] + '</function_calls>' }}"
        "{% endif %}"
        "{% if not loop.last %}"
        "{{ '<|im_end|>' + '\n' }}"
        "{% else %}"
        "{{ eos_token }}"
        "{% endif %}"
        "{% elif message['role'] == 'environment' %}"
        "{{ '<|im_start|>environment\n' + message['content'] + '<|im_end|>\n' }}"
        "{% endif %}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_rlzero": (
        "Solve the following problem step by step. "
        "The last line of your response should be the answer to the problem in form Answer: $Answer (without quotes) where $Answer is the answer to the problem."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        "{{ '\nRemember to put your answer on its own line after \"Answer:\"' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "olmo_thinker_code_rlzero": (
        "Solve the following code problem step by step. "
        f"The last part of your response should be the solution to the problem in form ```\npython\nCODE\n``` where CODE is the solution for the problem."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['content'] + '\n' }}"
        "{% if loop.last and add_generation_prompt %}"
        f"\nRemember to put your solution inside the ```\npython\nCODE\n``` tags"
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
    "r1_simple_chat_postpend_think_orz_style": (
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
        "{{ message['role'].capitalize() + ': You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\\\boxed{} tag. This is the problem: ' + message['content'] + '\n' }}"  # \\\\boxed{} is for jinja template escape
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
    "r1_simple_chat_postpend_think_tool_vllm": (
        "A conversation between User and Assistant. "
        "The User asks a question, and the Assistant solves it. "
        "The Assistant first thinks about the reasoning process in "
        "the mind and then provides the User with the answer. "
        "\n\n"
        "When given a question, the Assistant must conduct reasoning inside the <think> "
        "and </think> tags. During reasoning, the Assistant may write and execute python "
        "code using the <code> </code> tag, in order to solve the problem or verify the answer. "
        "Then the Assistant will get the stdout and stderr in the <output> and </output> tags. "
        "For example, the code could be\n"
        "<code>\n"
        "x, y = 1, 2\n"
        "result = x + y\n"
        "print(result)\n"
        "</code>\n"
        "or\n"
        "<code>\n"
        "import sympy as sp\n"
        "from sympy import Symbol\n"
        "x = Symbol('x')\n"
        "y = Symbol('y')\n"
        "solution = sp.solve(x**2 + y**2 - 1, (x, y))\n"
        "print(solution)\n"
        "</code>\n"
        "The Assistant will always `print` the result of the code execution in order to see it in the <output> tag. "
        "The Assistant may use the <code> </code> tag multiple times. "
        "When the Assistant is done reasoning, it should provide the answer inside the <answer> "
        "and </answer> tag."
        "\n\n"
        "{% for message in messages %}"
        "{{ '\n\n' if not loop.first else '' }}"
        "{{ message['role'].capitalize() + ': You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\\\boxed{} tag. This is the problem: ' + message['content'] + '\n' }}"  # \\\\boxed{} is for jinjia template escape
        "{% if loop.last and add_generation_prompt %}"
        "{{ 'Assistant: <think>' }}"
        "{% endif %}"
        "{% endfor %}"
    ),
}
# flake8: noqa


def get_tokenizer_simple_v1(tc: "TokenizerConfig"):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    return tokenizer


def get_tokenizer_tulu_v1(tc: "TokenizerConfig"):
    tokenizer = AutoTokenizer.from_pretrained(
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # only add if the pad token is not present already.
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens(
            {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}
        )
        assert num_added_tokens in [0, 1], (
            "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
        )
    elif isinstance(tokenizer, GPTNeoXTokenizerFast):
        # OLMo newer models use this tokenizer
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            assert tc.add_bos, "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
        # else, pythia / other models
        else:
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens <= 1, (
                "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
            )
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
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                tc.tokenizer_name_or_path, revision=tc.tokenizer_revision
            ).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.tokenizer_name_or_path}.")

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
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # only add if the pad token is not present already, or if the current one is set to eos_token_id.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens in [0, 1], (
                "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
            )
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            # OLMo newer models use this tokenizer
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
                assert tc.add_bos, (
                    "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence."
                )
            # else, pythia / other models
            else:
                num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
                assert num_added_tokens <= 1, (
                    "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
                )
        # NOTE: (Costa) I just commented the `OPTForCausalLM` because we are not likely to use it.
        # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        #     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
        elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "pad token and eos token matching causes issues in our setup."
    )

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name is None:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                tc.tokenizer_name_or_path, revision=tc.tokenizer_revision
            ).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.tokenizer_name_or_path}.")
    elif tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        raise ValueError(f"Could not find chat template for {tc.chat_template_name}.")

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


def get_tokenizer_tulu_v2_2(tc: "TokenizerConfig"):
    config = AutoConfig.from_pretrained(tc.tokenizer_name_or_path, revision=tc.tokenizer_revision)
    # @vwxyzjn: "olmo" handles both `olmo2` and `olmoe`.
    if "olmo" in config.model_type:
        if tc.chat_template_name is None:
            pass  # just assume the user knows what they're doing
        elif "olmo" in tc.chat_template_name:
            assert not tc.add_bos, "For newer OLMo chat templates, you must *not* run with `--add_bos`."
        else:
            assert tc.add_bos, "For OLMo, you must run with `--add_bos`."
        assert tc.use_fast, "For OLMo, you must use fast tokenizer."

    tokenizer = AutoTokenizer.from_pretrained(
        tc.tokenizer_name_or_path,
        revision=tc.tokenizer_revision,
        trust_remote_code=tc.trust_remote_code,
        use_fast=tc.use_fast,
    )
    # no default pad token for llama!
    # here we add all special tokens again, because the default ones are not in the special_tokens_map
    # only add if the pad token is not present already, or if the current one is set to eos_token_id.
    if tokenizer.pad_token_id is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens in [0, 1], (
                "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."
            )
        elif isinstance(tokenizer, GPTNeoXTokenizerFast):
            # OLMo newer models use this tokenizer
            if tokenizer.bos_token is None:
                tokenizer.bos_token = tokenizer.eos_token
                if "olmo" not in tc.chat_template_name:
                    assert tc.add_bos, (
                        "For OLMo with GPTNeoX, you must add bos token to the beginning of the input sequence "
                        "if using an older chat template."
                    )
            # else, pythia / other models
            else:
                num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
                assert num_added_tokens <= 1, (
                    "GPTNeoXTokenizer should only add one special token - the pad_token (or no tokens if already set in SFT)."
                )
        # NOTE: (Costa) I just commented the `OPTForCausalLM` because we are not likely to use it.
        # elif isinstance(tokenizer, GPT2Tokenizer) and isinstance(model, OPTForCausalLM):
        #     num_added_tokens = tokenizer.add_special_tokens({"unk_token": "<unk>"})
        elif isinstance(tokenizer, transformers.PreTrainedTokenizerFast):
            num_added_tokens = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            assert num_added_tokens == 1, "We detected no padding token but add_special_tokens did not add one."

    assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
        "pad token and eos token matching causes issues in our setup."
    )

    # set the tokenizer chat template to the training format
    # this will be used for encoding the training examples
    # and saved together with the tokenizer to be used later.
    if tc.chat_template_name in CHAT_TEMPLATES:
        tokenizer.chat_template = CHAT_TEMPLATES[tc.chat_template_name]
    else:
        try:
            tokenizer.chat_template = AutoTokenizer.from_pretrained(
                tc.tokenizer_name_or_path, revision=tc.tokenizer_revision
            ).chat_template
        except Exception:
            raise ValueError(f"Could not find chat template for {tc.tokenizer_name_or_path}.")

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
    "get_tokenizer_tulu_v2_2": get_tokenizer_tulu_v2_2,
}

DEFAULT_SFT_MESSAGES_KEY = "messages"
GROUND_TRUTHS_KEY = "ground_truth"
VERIFIER_SOURCE_KEY = "dataset"
RAW_PROMPT_KEY = "prompt"


@dataclass
class TokenizerConfig:
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    trust_remote_code: bool = False
    use_fast: bool = True
    chat_template_name: Optional[str] = None  # default to using the tokenizer chat template
    add_bos: bool = False
    get_tokenizer_fn: str = "get_tokenizer_tulu_v2_2"

    # for tracking purposes
    tokenizer_files_hash: Optional[List[str]] = None

    # backward compatibility to make sure script runs
    use_slow_tokenizer: bool = False  # completely ignored
    tokenizer_name: Optional[str] = None
    ground_truths_key: str = GROUND_TRUTHS_KEY
    """columns name for the ground truth"""
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY
    """columns name for the sft messages"""

    @cached_property
    def tokenizer(self):
        files_hash = get_files_hash_if_exists(
            self.tokenizer_name_or_path,
            self.tokenizer_revision,
            filenames=["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "vocab.json"],
        )
        self.tokenizer_files_hash = ",".join(files_hash)
        if self.tokenizer_name is not None and self.tokenizer_name_or_path is None:
            if self.tokenizer_name != self.tokenizer_name_or_path:
                raise ValueError(
                    f"tokenizer_name and tokenizer_name_or_path are different: {self.tokenizer_name=} != {self.tokenizer_name_or_path=},"
                    " you should use only `--tokenizer_name_or_path` in the future as `tokenizer_name` is deprecated."
                )
            self.tokenizer_name_or_path = self.tokenizer_name
        return GET_TOKENIZER_FN[self.get_tokenizer_fn](self)


# TODO: for testing, we should load the tokenizer from the sft / dpo / rl and make sure they are all the same.


# ----------------------------------------------------------------------------
# Dataset Transformation
# SFT dataset
INPUT_IDS_KEY = "input_ids"
ATTENTION_MASK_KEY = "attention_mask"
LABELS_KEY = "labels"
DATASET_ORIGIN_KEY = "dataset_source"  # just 'dataset' clashes with RLVR stuff (see VERIFIER_SOURCE_KEY)
TOKENIZED_SFT_DATASET_KEYS = [INPUT_IDS_KEY, ATTENTION_MASK_KEY, LABELS_KEY]
TOKENIZED_SFT_DATASET_KEYS_WITH_SOURCE = [INPUT_IDS_KEY, ATTENTION_MASK_KEY, LABELS_KEY, DATASET_ORIGIN_KEY]


def remove_dataset_source_field(dataset: Dataset) -> Dataset:
    """Remove dataset_source field from dataset if it exists.

    This should be called after statistics collection but before returning
    the final dataset to avoid storing unnecessary metadata in cached datasets.
    """
    if DATASET_ORIGIN_KEY in dataset.column_names:
        return dataset.remove_columns([DATASET_ORIGIN_KEY])
    return dataset


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

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    row.pop(RAW_PROMPT_KEY, None)
    return row


def sft_tokenize_mask_out_prompt_v1(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY
):
    """mask out the prompt tokens by manipulating labels"""
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
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


def last_turn_tulu_tokenize_and_truncate_v1(row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_seq_length: int):
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
    # mask all turns but the last for avoiding loss
    for message_idx, message in enumerate(messages):
        if message_idx < len(messages) - 1:
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
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
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


def preference_tulu_tokenize_and_truncate_v1_2(
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

    chosen_encoded = last_turn_tulu_tokenize_and_truncate_v1(
        {DEFAULT_SFT_MESSAGES_KEY: chosen_messages}, tokenizer, max_seq_length
    )
    rejected_encoded = last_turn_tulu_tokenize_and_truncate_v1(
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
    verifier_source_key: str = VERIFIER_SOURCE_KEY,
):
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    row[GROUND_TRUTHS_KEY] = row[ground_truths_key]
    row[VERIFIER_SOURCE_KEY] = row[verifier_source_key]
    # concatenate all the previous messages as <role>: <content>\n <role>: <content>\n ...
    row[RAW_PROMPT_KEY] = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt)
    return row


def rlvr_tokenize_v2(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
    ground_truths_key: str = GROUND_TRUTHS_KEY,
    verifier_source_key: str = VERIFIER_SOURCE_KEY,
):
    if len(row[sft_messages_key]) == 1:
        prompt = row[sft_messages_key]
    else:
        prompt = row[sft_messages_key][:-1]
    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    row[INPUT_IDS_KEY] = tokenizer.apply_chat_template(row[sft_messages_key])
    # weird issue with qwen: sometimes the padding token ends up in the input ids?
    # ill look into this more later, for now this guard should be enough
    if tokenizer.pad_token_id in row[INPUT_IDS_KEY]:
        row[INPUT_IDS_KEY] = [x for x in row[INPUT_IDS_KEY] if x != tokenizer.pad_token_id]
    if tokenizer.pad_token_id in row[INPUT_IDS_PROMPT_KEY]:
        row[INPUT_IDS_PROMPT_KEY] = [x for x in row[INPUT_IDS_PROMPT_KEY] if x != tokenizer.pad_token_id]
    row[ATTENTION_MASK_KEY] = [1] * len(row[INPUT_IDS_KEY])
    labels = copy.deepcopy(row[INPUT_IDS_KEY])
    row[LABELS_KEY] = labels
    row[GROUND_TRUTHS_KEY] = row[ground_truths_key]
    row[VERIFIER_SOURCE_KEY] = row[verifier_source_key]
    # concatenate all the previous messages as <role>: <content>\n <role>: <content>\n ...
    # row[DEFAULT_SFT_MESSAGES_KEY] = prompt
    # concatenate all the previous messages as <role>: <content>\n <role>: <content>\n ...
    row[RAW_PROMPT_KEY] = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt)
    # some basic transformations:
    # if ground truths is a string, make it a list
    if isinstance(row[ground_truths_key], str):
        row[ground_truths_key] = [row[ground_truths_key]]
    # if dataset source is a string, make it a list
    if isinstance(row[verifier_source_key], str):
        row[verifier_source_key] = [row[verifier_source_key]]
    # drop the messages field as it often causes issues.
    row.pop(sft_messages_key)
    return row


def rlvr_tokenize_v3(
    row: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    sft_messages_key: str = DEFAULT_SFT_MESSAGES_KEY,
    ground_truths_key: str = GROUND_TRUTHS_KEY,
    verifier_source_key: str = VERIFIER_SOURCE_KEY,
    system_prompt_override: Optional[str] = None,
):
    prompt = row.pop(sft_messages_key)
    assert len(prompt) > 0, "Empty prompt in dataset"
    # if the prompt has multiple messages, make sure we don't end in an assistant message.
    if len(prompt) > 1 and prompt[-1]["role"] == "assistant":
        prompt = prompt[:-1]
    # override the system prompt if we have a new one provided.
    if system_prompt_override:
        if prompt[0]["role"] == "system":
            del prompt[0]
        prompt = [{"role": "system", "content": system_prompt_override}] + prompt

    row[INPUT_IDS_PROMPT_KEY] = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
    if tokenizer.pad_token_id in row[INPUT_IDS_PROMPT_KEY]:
        row[INPUT_IDS_PROMPT_KEY] = [x for x in row[INPUT_IDS_PROMPT_KEY] if x != tokenizer.pad_token_id]
    row[GROUND_TRUTHS_KEY] = row[ground_truths_key]
    row[VERIFIER_SOURCE_KEY] = row[verifier_source_key]
    # concatenate all the previous messages as <role>: <content>\n <role>: <content>\n ...
    # row[DEFAULT_SFT_MESSAGES_KEY] = prompt
    # concatenate all the previous messages as <role>: <content>\n <role>: <content>\n ...
    row[RAW_PROMPT_KEY] = "\n".join(f"{msg['role']}: {msg['content']}" for msg in prompt)
    # some basic transformations:
    # if ground truths is a string, make it a list
    if isinstance(row[ground_truths_key], str):
        row[ground_truths_key] = [row[ground_truths_key]]
    # if dataset source is a string, make it a list
    if isinstance(row[verifier_source_key], str):
        row[verifier_source_key] = [row[verifier_source_key]]
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


def rlvr_max_length_filter_v2(
    row: Dict[str, Any], tokenizer: PreTrainedTokenizer, max_prompt_token_length: Optional[int] = None
):
    if max_prompt_token_length is None:
        return True
    return len(row[INPUT_IDS_PROMPT_KEY]) <= max_prompt_token_length


TRANSFORM_FNS = {
    "sft_tokenize_v1": (sft_tokenize_v1, "map"),
    "sft_tokenize_mask_out_prompt_v1": (sft_tokenize_mask_out_prompt_v1, "map"),
    "sft_filter_v1": (sft_filter_v1, "filter"),
    "sft_tulu_tokenize_and_truncate_v1": (sft_tulu_tokenize_and_truncate_v1, "map"),
    "sft_tulu_filter_v1": (sft_tulu_filter_v1, "filter"),
    "preference_tokenize_v1": (preference_tokenize_v1, "map"),
    "preference_filter_v1": (preference_filter_v1, "filter"),
    "preference_tulu_tokenize_and_truncate_v1": (preference_tulu_tokenize_and_truncate_v1_2, "map"),
    "preference_tulu_filter_v1": (preference_tulu_filter_v1, "filter"),
    "rlvr_tokenize_v1": (rlvr_tokenize_v3, "map"),
    "rlvr_max_length_filter_v1": (rlvr_max_length_filter_v2, "filter"),
}


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
            max_length_chosen = max(max_length_chosen, len(batch[i][CHOSEN_INPUT_IDS_KEY]))
            max_length_rejected = max(max_length_rejected, len(batch[i][REJECTED_INPUT_IDS_KEY]))
        max_length = max(max_length_chosen, max_length_rejected)
        assert max_length > 0, "the dataset is empty"

        # Initialize lists to store padded sequences and attention masks
        padded_sequences_chosen = []
        padded_sequences_rejected = []

        for i in range(len(batch)):
            # Calculate padding length
            pad_length_chosen = max_length - len(batch[i][CHOSEN_INPUT_IDS_KEY])
            pad_length_rejected = max_length - len(batch[i][REJECTED_INPUT_IDS_KEY])

            # Pad from the right
            padding_chosen = [self.pad_token_id] * pad_length_chosen
            padding_rejected = [self.pad_token_id] * pad_length_rejected
            padded_sequence_chosen = batch[i][CHOSEN_INPUT_IDS_KEY] + padding_chosen
            padded_sequence_rejected = batch[i][REJECTED_INPUT_IDS_KEY] + padding_rejected
            padded_sequences_chosen.append(padded_sequence_chosen)
            padded_sequences_rejected.append(padded_sequence_rejected)

        # Convert to tensors
        padded_sequences_chosen = torch.tensor(padded_sequences_chosen)
        padded_sequences_rejected = torch.tensor(padded_sequences_rejected)

        return {CHOSEN_INPUT_IDS_KEY: padded_sequences_chosen, REJECTED_INPUT_IDS_KEY: padded_sequences_rejected}


# ----------------------------------------------------------------------------
# Dataset Configuration and Caching
@dataclass
class DatasetConfig:
    dataset_name: str
    dataset_split: str
    dataset_revision: str
    dataset_range: Optional[int] = None
    transform_fn: List[str] = field(default_factory=list)
    transform_fn_args: List[Dict[str, Any]] = field(default_factory=list)
    target_columns: Optional[List[str]] = None
    dataset_config_seed: int = 42

    # for tracking purposes
    dataset_commit_hash: Optional[str] = None
    frac_or_num_samples: Optional[Union[int, float]] = None
    original_dataset_size: Optional[int] = None
    is_upsampled: bool = False

    def __post_init__(self):
        # if the file exists locally, use the local file
        if os.path.exists(self.dataset_name) and self.dataset_name.endswith(".jsonl"):
            assert self.dataset_split == "train", "Only train split is supported for local jsonl files."
            self.dataset = load_dataset(
                "json", data_files=self.dataset_name, split=self.dataset_split, num_proc=max_num_processes()
            )
        elif os.path.exists(self.dataset_name) and self.dataset_name.endswith(".parquet"):
            assert self.dataset_split == "train", "Only train split is supported for local parquet files."
            self.dataset = load_dataset(
                "parquet", data_files=self.dataset_name, split=self.dataset_split, num_proc=max_num_processes()
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
                num_proc=max_num_processes(),
            )
        if self.dataset_range is None:
            dataset_range = len(self.dataset)
            self.update_range(dataset_range)

    def update_range(self, dataset_range: int):
        self.dataset_range = dataset_range
        original_size = len(self.dataset)
        self.original_dataset_size = original_size

        self.dataset = self.select_samples(self.dataset_range)
        self.is_upsampled = dataset_range > original_size

    def select_samples(self, target_size: int):
        """Upsample dataset to target_size by repeating samples."""
        original_size = len(self.dataset)

        # Calculate how many full repeats and how many extra samples
        full_repeats = target_size // original_size
        extra_samples = target_size % original_size

        # Create indices for upsampling
        indices = []

        # Add full repeats
        for _ in range(full_repeats):
            indices.extend(range(original_size))

        # Add randomly sampled extra samples
        if extra_samples > 0:
            # Use numpy for reproducible random sampling
            rng = np.random.RandomState(self.dataset_config_seed)
            extra_indices = rng.choice(original_size, size=extra_samples, replace=False)
            indices.extend(extra_indices.tolist())

        if target_size > original_size:
            print(
                f"Upsampling dataset {self.dataset_name} from {original_size} to {target_size} samples "
                f"({full_repeats} full repeats + {extra_samples} random samples)"
            )

        return self.dataset.select(indices)


def get_dataset_v1(dc: DatasetConfig, tc: TokenizerConfig):
    assert len(dc.transform_fn) == len(dc.transform_fn_args), (
        f"transform_fn and transform_fn_args must have the same length: {dc.transform_fn=} != {dc.transform_fn_args=}"
    )
    # beaker specific logic; we may get assigned 15.5 CPU, so we convert it to float then int
    num_proc = int(float(os.environ.get("BEAKER_ASSIGNED_CPU_COUNT", multiprocessing.cpu_count())))

    tokenizer = tc.tokenizer
    dataset = dc.dataset

    # Add dataset source field to track origin after shuffling
    dataset = dataset.map(
        lambda example: {**example, DATASET_ORIGIN_KEY: dc.dataset_name},
        num_proc=num_proc,
        desc=f"Adding dataset source field for {dc.dataset_name}",
    )
    for fn_name, fn_args in zip(dc.transform_fn, dc.transform_fn_args):
        fn, fn_type = TRANSFORM_FNS[fn_name]
        # always pass in tokenizer and other args if needed
        fn_kwargs = {"tokenizer": tokenizer}
        fn_kwargs.update(fn_args)

        # perform the transformation
        target_columns = dataset.column_names if dc.target_columns is None else dc.target_columns
        # Always preserve dataset_source if it exists
        if DATASET_ORIGIN_KEY in dataset.column_names and DATASET_ORIGIN_KEY not in target_columns:
            target_columns = target_columns + [DATASET_ORIGIN_KEY]

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


def compute_config_hash(dcs: List[DatasetConfig], tc: TokenizerConfig) -> str:
    """Compute a deterministic hash of both configs for caching."""
    dc_dicts = [{k: v for k, v in asdict(dc).items() if v is not None} for dc in dcs]
    tc_dict = {k: v for k, v in asdict(tc).items() if v is not None}
    combined_dict = {"dataset_configs": dc_dicts, "tokenizer_config": tc_dict}
    config_str = json.dumps(combined_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:10]


class DatasetTransformationCache:
    def __init__(self, config_hash: str, hf_entity: Optional[str] = None):
        self.config_hash = config_hash
        self.hf_entity = hf_entity or hf_whoami()["name"]

    def load_or_transform_dataset(
        self, dcs: List[DatasetConfig], tc: TokenizerConfig, dataset_skip_cache: bool = False
    ) -> Dataset:
        """Load dataset from cache if it exists, otherwise transform and cache it."""
        repo_name = f"{self.hf_entity}/dataset-mix-cached"

        # NOTE: the cached dataset is always train split
        DEFAULT_SPLIT_FOR_CACHED_DATASET = "train"

        # Check if the revision exists
        if revision_exists(repo_name, self.config_hash, repo_type="dataset"):
            print(f"✅ Found cached dataset at https://huggingface.co/datasets/{repo_name}/tree/{self.config_hash}")
            if dataset_skip_cache:
                print("dataset_skip_cache is True, so we will not load the dataset from cache")
            else:
                # Use the split from the first dataset config as default
                return load_dataset(
                    repo_name,
                    split=DEFAULT_SPLIT_FOR_CACHED_DATASET,
                    revision=self.config_hash,
                    num_proc=max_num_processes(),
                )

        print(f"Cache not found, transforming datasets...")

        # Transform each dataset
        transformed_datasets = []
        for dc in dcs:
            dataset = get_dataset_v1(dc, tc)
            transformed_datasets.append(dataset)

        # Combine datasets
        combined_dataset = concatenate_datasets(transformed_datasets)
        if dataset_skip_cache:
            return combined_dataset

        # Push to hub with config hash as revision
        combined_dataset.push_to_hub(
            repo_name,
            private=True,
            revision=self.config_hash,
            commit_message=f"Cache combined dataset with configs hash: {self.config_hash}",
        )
        print(f"🚀 Pushed transformed dataset to https://huggingface.co/datasets/{repo_name}/tree/{self.config_hash}")

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
        model_card.push_to_hub(repo_name, repo_type="dataset", revision=self.config_hash)

        # NOTE: Load the dataset again to make sure it's downloaded to the HF cache
        print(f"✅ Found cached dataset at https://huggingface.co/datasets/{repo_name}/tree/{self.config_hash}")
        return load_dataset(
            repo_name, split=DEFAULT_SPLIT_FOR_CACHED_DATASET, revision=self.config_hash, num_proc=max_num_processes()
        )


class LocalDatasetTransformationCache:
    def __init__(self, config_hash: str, dataset_local_cache_dir: str):
        """Initialize the local cache with a directory path."""
        self.config_hash = config_hash
        self.dataset_local_cache_dir = dataset_local_cache_dir
        os.makedirs(dataset_local_cache_dir, exist_ok=True)

    def get_cache_path(self) -> str:
        """Get the path to the cached dataset."""
        return os.path.join(self.dataset_local_cache_dir, self.config_hash)

    def save_config(self, config_hash: str, dcs: List[DatasetConfig], tc: TokenizerConfig):
        """Save the configuration to a JSON file."""
        config_path = os.path.join(self.get_cache_path(), "config.json")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        config_dict = {
            "tokenizer_config": asdict(tc),
            "dataset_configs": [asdict(dc) for dc in dcs],
            "config_hash": config_hash,
        }
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

    def load_or_transform_dataset(
        self, dcs: List[DatasetConfig], tc: TokenizerConfig, dataset_skip_cache: bool = False
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """Load dataset from local cache if it exists, otherwise transform and cache it locally."""
        cache_path = self.get_cache_path()

        # Check if the cache exists
        if os.path.exists(cache_path) and not dataset_skip_cache:
            print(f"✅ Found cached dataset at {cache_path}")
            dataset = Dataset.load_from_disk(cache_path, keep_in_memory=True)
            # Load statistics from cache if available
            stats_path = os.path.join(cache_path, "dataset_statistics.json")
            if os.path.exists(stats_path):
                with open(stats_path, "r") as f:
                    statistics = json.load(f)
                return dataset, statistics
            else:
                # Return empty statistics if not cached
                return dataset, {"per_dataset_stats": [], "dataset_order": []}

        print(f"Cache not found or invalid, transforming datasets...")

        # Transform each dataset and collect statistics
        transformed_datasets = []
        dataset_statistics = []
        dataset_order = []

        for dc in dcs:
            # Get initial dataset info
            initial_size = len(dc.dataset) if dc.dataset else 0

            dataset = get_dataset_v1(dc, tc)
            transformed_datasets.append(dataset)

            # Collect statistics for this dataset
            stats = {
                "dataset_name": dc.dataset_name,
                "dataset_split": dc.dataset_split,
                "initial_instances": initial_size,
                "final_instances": len(dataset),
                "instances_filtered": initial_size - len(dataset),
                "frac_or_num_samples": dc.frac_or_num_samples,
                "original_dataset_size": dc.original_dataset_size,
                "is_upsampled": dc.is_upsampled,
                "upsampling_factor": dc.dataset_range / dc.original_dataset_size
                if dc.original_dataset_size and dc.original_dataset_size > 0
                else 1.0,
            }

            # Count tokens if the dataset has been tokenized
            if INPUT_IDS_KEY in dataset.column_names:
                total_tokens = 0
                trainable_tokens = 0

                def count_tokens(sample):
                    token_count = len(sample[INPUT_IDS_KEY])
                    trainable_tokens = sum(1 for label in sample[LABELS_KEY] if label != -100)
                    return {"token_count": token_count, "label_token_count": trainable_tokens}

                token_count_dataset = dataset.map(count_tokens, batched=False)
                total_tokens = sum(token_count_dataset["token_count"])
                trainable_tokens = sum(token_count_dataset["label_token_count"])
                stats["total_tokens"] = total_tokens
                stats["trainable_tokens"] = trainable_tokens
                stats["avg_tokens_per_instance"] = total_tokens / len(dataset) if len(dataset) > 0 else 0

            dataset_statistics.append(stats)
            dataset_order.append(dc.dataset_name)

        # Combine datasets
        combined_dataset = concatenate_datasets(transformed_datasets)

        # Prepare return statistics
        all_statistics = {"per_dataset_stats": dataset_statistics, "dataset_order": dataset_order}

        if dataset_skip_cache:
            return combined_dataset, all_statistics

        # Save to local cache
        combined_dataset.save_to_disk(cache_path)
        self.save_config(self.config_hash, dcs, tc)

        # Save statistics to cache
        stats_path = os.path.join(cache_path, "dataset_statistics.json")
        with open(stats_path, "w") as f:
            json.dump(all_statistics, f, indent=2)

        print(f"🚀 Saved transformed dataset to {cache_path}")
        print(f"✅ Found cached dataset at {cache_path}")

        loaded_dataset = Dataset.load_from_disk(cache_path, keep_in_memory=True)
        return loaded_dataset, all_statistics


def load_dataset_configs(
    dataset_mixer_list: List[str],
    dataset_mixer_list_splits: List[str],
    dataset_transform_fn: List[str],
    transform_fn_args: List[Dict[str, Any]],
    target_columns: Optional[List[str]] = None,
    dataset_config_seed: int = 42,
) -> List[DatasetConfig]:
    dcs = []
    if len(dataset_mixer_list_splits) == 1:
        print("by default, we will use the same split for all datasets")
        dataset_mixer_list_splits = [dataset_mixer_list_splits[0]] * len(dataset_mixer_list)
    else:
        if len(dataset_mixer_list_splits) != len(dataset_mixer_list):
            raise ValueError(
                f"dataset_mixer_list_splits length must be the same as dataset_mixer_list: {len(dataset_mixer_list_splits)=} != {len(dataset_mixer_list)=}"
            )
    assert len(dataset_mixer_list) % 2 == 0, f"Data mixer list length is not even: {dataset_mixer_list}"
    for i in range(0, len(dataset_mixer_list), 2):
        dataset_name = dataset_mixer_list[i]
        frac_or_num_samples = dataset_mixer_list[i + 1]
        if "." in frac_or_num_samples:
            frac_or_num_samples = float(frac_or_num_samples)
        else:
            frac_or_num_samples = int(frac_or_num_samples)
        # Uses dataset_mixer_list_splits[i] where i increments by 2 (0, 2, 4...). This works because
        # all current usage provides a single split that gets replicated to len(dataset_mixer_list).
        # If different splits per dataset are needed, use dataset_mixer_list_splits[i // 2] instead.
        assert i % 2 == 0, f"Index {i} must be even"
        assert i < len(dataset_mixer_list_splits), (
            f"Index {i} out of bounds for dataset_mixer_list_splits of length {len(dataset_mixer_list_splits)}"
        )
        dataset_config = DatasetConfig(
            dataset_name=dataset_name,
            dataset_split=dataset_mixer_list_splits[i],
            dataset_revision="main",
            transform_fn=dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            target_columns=target_columns,
            frac_or_num_samples=frac_or_num_samples,
            dataset_config_seed=dataset_config_seed,
        )

        original_size = len(dataset_config.dataset)
        if isinstance(frac_or_num_samples, int) and frac_or_num_samples > original_size:
            new_range = frac_or_num_samples
        elif isinstance(frac_or_num_samples, float):
            new_range = int(frac_or_num_samples * original_size)
        else:
            new_range = int(frac_or_num_samples)

        print(f"Dataset {dataset_name}: {original_size} -> {new_range} samples (factor: {frac_or_num_samples})")
        dataset_config.update_range(new_range)
        dcs.append(dataset_config)
    return dcs


def get_cached_dataset_tulu_with_statistics(
    dataset_mixer_list: List[str],
    dataset_mixer_list_splits: List[str],
    tc: TokenizerConfig,
    dataset_transform_fn: List[str],
    transform_fn_args: List[Dict[str, Any]],
    target_columns: Optional[List[str]] = None,
    dataset_cache_mode: Literal["hf", "local"] = "local",
    dataset_config_hash: Optional[str] = None,
    hf_entity: Optional[str] = None,
    dataset_local_cache_dir: str = "local_dataset_cache",
    dataset_skip_cache: bool = False,
    drop_dataset_source: bool = True,
    dataset_config_seed: int = 42,
    system_prompt_override: Optional[str] = None,
) -> Union[Dataset, Tuple[Dataset, Dict[str, Any]]]:
    if dataset_config_hash is None:
        dcs = load_dataset_configs(
            dataset_mixer_list,
            dataset_mixer_list_splits,
            dataset_transform_fn,
            transform_fn_args,
            target_columns,
            dataset_config_seed,
        )
        dataset_config_hash = compute_config_hash(dcs, tc)
    else:
        dcs = []
    if dataset_cache_mode == "local":
        cache = LocalDatasetTransformationCache(
            config_hash=dataset_config_hash, dataset_local_cache_dir=dataset_local_cache_dir
        )
    elif dataset_cache_mode == "hf":
        cache = DatasetTransformationCache(config_hash=dataset_config_hash, hf_entity=hf_entity)

    dataset, statistics = cache.load_or_transform_dataset(dcs, tc, dataset_skip_cache=dataset_skip_cache)

    if drop_dataset_source:
        dataset = remove_dataset_source_field(dataset)

    return dataset, statistics


def get_cached_dataset_tulu(
    dataset_mixer_list: List[str],
    dataset_mixer_list_splits: List[str],
    tc: TokenizerConfig,
    dataset_transform_fn: List[str],
    transform_fn_args: List[Dict[str, Any]],
    target_columns: Optional[List[str]] = None,
    dataset_cache_mode: Literal["hf", "local"] = "local",
    dataset_config_hash: Optional[str] = None,
    hf_entity: Optional[str] = None,
    dataset_local_cache_dir: str = "local_dataset_cache",
    dataset_skip_cache: bool = False,
    dataset_config_seed: int = 42,
    system_prompt_override: Optional[str] = None,
) -> Dataset:
    return get_cached_dataset_tulu_with_statistics(
        dataset_mixer_list=dataset_mixer_list,
        dataset_mixer_list_splits=dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        target_columns=target_columns,
        dataset_cache_mode=dataset_cache_mode,
        dataset_config_hash=dataset_config_hash,
        hf_entity=hf_entity,
        dataset_local_cache_dir=dataset_local_cache_dir,
        dataset_skip_cache=dataset_skip_cache,
        drop_dataset_source=True,
        dataset_config_seed=dataset_config_seed,
        system_prompt_override=system_prompt_override,
    )[0]
