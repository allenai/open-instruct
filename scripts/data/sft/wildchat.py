import argparse

from datasets import load_dataset

import open_instruct.utils as open_instruct_utils
from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WildChat dataset and optionally upload to Hugging Face Hub.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )
    parser.add_argument(
        "--converted_dataset_name", type=str, default=None, help="Name of the converted dataset on Hugging Face Hub."
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

    ds = load_dataset("allenai/WildChat-1M-Full", num_proc=open_instruct_utils.max_num_processes())
    gpt4_subset = ds.filter(lambda example: "gpt-4" in example["model"], num_proc=16)
    gpt35_subset = ds.filter(lambda example: "gpt-3.5" in example["model"], num_proc=16)
    assert len(gpt4_subset["train"]) + len(gpt35_subset["train"]) == len(
        ds["train"]
    ), "There are some examples not using either gpt-4 or gpt-3.5"

    conversion_func = lambda example: {
        "messages": [{"role": it["role"], "content": it["content"]} for it in example["conversation"]]
    }

    gpt4_subset_readme_content = (
        "This is a converted version of the gpt-4 subset of the WildChat dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/wildchat.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/allenai/WildChat-1M-Full) "
        "for more information about this dataset and the license."
    )

    gpt35_subset_readme_content = (
        "This is a converted version of the gpt-3.5 subset of the WildChat dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/wildchat.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original dataset](https://huggingface.co/datasets/allenai/WildChat-1M-Full) "
        "for more information about this dataset and the license."
    )

    convert_sft_dataset(
        ds=gpt4_subset,
        hf_dataset_id=None,
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="wildchat_gpt4_converted"
        if not args.converted_dataset_name
        else args.converted_dataset_name + "_gpt4",
        local_save_dir=args.local_save_dir,
        readme_content=gpt4_subset_readme_content,
    )

    convert_sft_dataset(
        ds=gpt35_subset,
        hf_dataset_id=None,
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="wildchat_gpt35_converted"
        if not args.converted_dataset_name
        else args.converted_dataset_name + "_gpt35",
        local_save_dir=args.local_save_dir,
        readme_content=gpt35_subset_readme_content,
    )
