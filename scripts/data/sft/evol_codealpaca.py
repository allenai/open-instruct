import argparse
import os
from collections import defaultdict
from typing import List, Optional

from scripts.data.sft.utils import convert_sft_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process the Evol-CodeAlpaca dataset and optionally upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Upload the dataset to Hugging Face Hub",
    )
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )
    parser.add_argument(
        "--converted_dataset_name",
        type=str,
        default="evol_codealpaca_converted",
        help="Name of the converted dataset on Hugging Face Hub.",
    )
    parser.add_argument(
        "--local_save_dir",
        type=str,
        default=None,
        help="Local directory to save the dataset (if not provided, does not save locally)",
    )
    parser.add_argument(
        "--apply_keyword_filters",
        action="store_true",
        help=(
            "Apply keyword filters to the dataset. "
            "Currently filters out conversations with OpenAI, ChatGPT, BingChat, etc."
        ),
    )
    parser.add_argument(
        "--apply_empty_message_filters",
        action="store_true",
        help="Apply empty message filters to the dataset.",
    )
    args = parser.parse_args()
    
    conversion_func = lambda example: {
        "messages": [
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["response"]}
        ]
    }
    
    readme_content = (
        "This is a converted version of the Evol-CodeAlpaca dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/evol_codealpaca.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "The original Evol-CodeAlpaca dataset can be found [here](https://huggingface.co/datasets/theblackcat102/evol-codealpaca-v1). "
        "Our conversion is based on the decontaminated version of the dataset, "
        "[`ise-uiuc/Magicoder-Evol-Instruct-110K`](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K).\n"
        "Please refer to the [Evol-CodeAlpaca repo](https://github.com/ise-uiuc/Evol-CodeAlpaca) and "
        "[Magicoder-Evol-Instruct-110K repo](https://github.com/ise-uiuc/Magicoder-Evol-Instruct-110K) "
        "for more information about this dataset and the license."
    )
    
    convert_sft_dataset(
        ds=None,
        hf_dataset_id="ise-uiuc/Magicoder-Evol-Instruct-110K",
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name=args.converted_dataset_name,
        local_save_dir=args.local_save_dir,
        readme_content=readme_content,
    )
    
