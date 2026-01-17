import argparse

from datasets import load_dataset

import open_instruct.utils as open_instruct_utils
from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process WebInstruct dataset and optionally upload to Hugging Face Hub."
    )
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

    ds = load_dataset("TIGER-Lab/WebInstructSub", num_proc=open_instruct_utils.max_num_processes())
    stack_exchange_sub = ds.filter(lambda example: example["source"] in ["mathstackexchange", "stackexchange"])
    socratic_sub = ds.filter(lambda example: example["source"] == "socratic")
    assert len(stack_exchange_sub["train"]) + len(socratic_sub["train"]) == len(
        ds["train"]
    ), "There are some examples that are not in either stack exchange or socratic"

    conversion_func = lambda example: {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }

    stack_exchange_sub_readme_content = (
        "This is a converted version of the stack exchange subset of the WebInstruct dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/web_instruct.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "We split the WebInstruct dataset into two subsets: stack exchange and socratic, and converted each separately. "
        "This is because the different license requirements for each."
        "Please refer to the [original dataset](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) "
        "for more information about this dataset and the license."
    )

    socratic_sub_readme_content = (
        "This is a converted version of the socratic subset of the WebInstruct dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/web_instruct.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "We split the WebInstruct dataset into two subsets: stack exchange and socratic, and converted each separately. "
        "This is because the different license requirements for each."
        "Please refer to the [original dataset](https://huggingface.co/datasets/TIGER-Lab/WebInstructSub) "
        "for more information about this dataset and the license."
    )

    convert_sft_dataset(
        ds=stack_exchange_sub,
        hf_dataset_id=None,
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="webinstruct_se_sub_converted"
        if not args.converted_dataset_name
        else args.converted_dataset_name + "_se_sub",
        local_save_dir=args.local_save_dir,
        readme_content=stack_exchange_sub_readme_content,
    )

    convert_sft_dataset(
        ds=socratic_sub,
        hf_dataset_id=None,
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="webinstruct_socratic_sub_converted"
        if not args.converted_dataset_name
        else args.converted_dataset_name + "_socratic_sub",
        local_save_dir=args.local_save_dir,
        readme_content=socratic_sub_readme_content,
    )
