import argparse
import re

from datasets import load_dataset

import open_instruct.utils as open_instruct_utils
from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process lmsys_chat_1m dataset and optionally upload to Hugging Face Hub."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )
    parser.add_argument(
        "--converted_dataset_name",
        type=str,
        default="lmsys_chat_1m_converted",
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
        "--apply_empty_message_filters", action="store_true", help="Apply empty message filters to the dataset."
    )
    parser.add_argument(
        "--model_name_regex",
        type=str,
        default="gpt-4",
        help="A regular expression. We will only keep instances where the model name contains this regular expression.",
    )
    args = parser.parse_args()

    readme_content = (
        "This is a converted version of the lmsys_chat_1m dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/lmsys_chat.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- model_name_regex: {args.model_name_regex}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) "
        "for more information about this dataset and the license."
    )

    ds = load_dataset("lmsys/lmsys-chat-1m", num_proc=open_instruct_utils.max_num_processes())
    if args.model_name_regex:
        ds = ds.filter(lambda example: re.search(args.model_name_regex, example["model"]))
        print(f"Dataset size after model name regex: {ds.num_rows}")

    convert_sft_dataset(
        ds=ds,
        hf_dataset_id=None,
        convert_fn=lambda x: {"messages": x["conversation"]},
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name=args.converted_dataset_name,
        local_save_dir=args.local_save_dir,
        readme_content=readme_content,
    )
