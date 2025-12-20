import os
import re
import tempfile
from collections.abc import Callable

import datasets
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

import open_instruct.utils as open_instruct_utils


def should_be_filtered_by_keyword(example, verbose=False):
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
        "LAION",
        "Open Assistant",
        "OpenAssistant",
        # Following keywords have more other meanings in context,
        # and they are not commonly used in our current datasets,
        # so we don't filter them by default.
        # "BARD",
        # "PaLM",
        # "Gemini",
        # "Gemma",
        # "Google AI",
        # "Anthropic",
        # "Claude",
        # "LLaMA",
        # "Meta AI",
        # "Mixtral",
        # "Mistral",
    ]
    for message in example["messages"]:
        if message["role"] != "assistant":
            continue
        # search for any of the filter strings in the content, case insensitive
        if re.search(r"\b(" + "|".join([s.lower() for s in filter_strings]) + r")\b", message["content"].lower()):
            if verbose:
                print("--------------------------------")
                print("Instance is filtered out because of the following message:")
                print(message["content"])
                print("It contains the following string(s):")
                for s in filter_strings:
                    if re.search(r"\b" + s.lower() + r"\b", message["content"].lower()):
                        print(s)
            return True
    return False


def should_be_filtered_by_empty_message(example, verbose=False):
    # we filter out conversations that contain empty messages
    for message in example["messages"]:
        if message["content"] == None or len(message["content"].strip()) == 0:
            if verbose:
                print("--------------------------------")
                print("Instance is filtered out because of an empty message:")
                print(message["content"])
            return True
    return False


def convert_sft_dataset(
    ds: Dataset = None,
    hf_dataset_id: str = None,
    apply_keyword_filters: bool = True,
    apply_empty_message_filters: bool = True,
    push_to_hub: bool = False,
    hf_entity: str | None = None,
    converted_dataset_name: str | None = None,
    local_save_dir: str | None = None,
    convert_fn: Callable | None = None,
    readme_content: str | None = None,
    num_proc: int = 16,
):
    datasets.disable_caching()  # force the latest dataset to be downloaded and reprocessed again

    assert ds is not None or hf_dataset_id is not None, "Either ds or hf_dataset_id must be provided."

    if ds is None:
        ds = load_dataset(hf_dataset_id, num_proc=open_instruct_utils.max_num_processes())

    if convert_fn:
        print("Converting dataset to messages format...")
        ds = ds.map(convert_fn, num_proc=num_proc)
    else:
        print("No convert_fn provided, skipping the conversion step.")

    if apply_keyword_filters:
        print("Filtering dataset by keywords...")
        ds = ds.filter(lambda example: not should_be_filtered_by_keyword(example), num_proc=num_proc)

    if apply_empty_message_filters:
        print("Filtering dataset by empty messages...")
        ds = ds.filter(lambda example: not should_be_filtered_by_empty_message(example), num_proc=num_proc)

    print(f"Dataset size after conversion: {ds.num_rows}")

    if push_to_hub:
        if hf_entity:
            full_name = f"{hf_entity}/{converted_dataset_name}"
        else:
            full_name = converted_dataset_name
        print(f"Pushing dataset to Hub: {full_name}")
        ds.push_to_hub(full_name, private=True)

        if readme_content:
            create_hf_readme(full_name, readme_content)

    if local_save_dir:
        print(f"Saving dataset locally to {local_save_dir}")
        ds.save_to_disk(local_save_dir)
        if readme_content:
            with open(os.path.join(local_save_dir, "README.md"), "w") as f:
                f.write(readme_content)

    return ds


def create_hf_readme(repo_id, content):
    """Creates a temporary README.md file and uploads it to a Hugging Face repository."""

    with tempfile.TemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, "README.md"), "w") as f:
            f.write(content)
        api = HfApi()
        api.upload_file(
            path_or_fileobj=os.path.join(temp_dir, "README.md"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
