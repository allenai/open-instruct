import argparse
import json
import os
import tempfile
from urllib.request import urlretrieve

from datasets import Dataset

from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ShareGPT dataset and optionally upload to Hugging Face Hub.")
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
        default="sharegpt_converted",
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
    args = parser.parse_args()

    readme_content = (
        "This is a converted version of the ShareGPT dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/sharegpt.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) "
        "for more information about this dataset and the license."
    )

    data = []
    # here we manually download the dataset because hf load_dataset errors out when loading from data files directly.
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Downloading ShareGPT files...")
        urlretrieve(
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json",
            os.path.join(temp_dir, "sg_90k_part1_html_cleaned.json"),
        )
        urlretrieve(
            "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json",
            os.path.join(temp_dir, "sg_90k_part2_html_cleaned.json"),
        )

        with open(os.path.join(temp_dir, "sg_90k_part1_html_cleaned.json")) as f:
            data.extend(json.load(f))
        with open(os.path.join(temp_dir, "sg_90k_part2_html_cleaned.json")) as f:
            data.extend(json.load(f))

    data = [d for d in data if "conversations" in d and len(d["conversations"]) > 0]

    role_map = {
        "user": "user",
        "human": "user",
        "bard": "assistant",
        "bing": "assistant",
        "chatgpt": "assistant",
        "gpt": "assistant",
        "system": "system",
    }
    data = [
        {"id": d["id"], "messages": [{"role": role_map[c["from"]], "content": c["value"]} for c in d["conversations"]]}
        for d in data
    ]

    ds = Dataset.from_list(data)
    convert_sft_dataset(
        ds=ds,
        hf_dataset_id=None,
        convert_fn=None,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name=args.converted_dataset_name,
        local_save_dir=args.local_save_dir,
        readme_content=readme_content,
    )
