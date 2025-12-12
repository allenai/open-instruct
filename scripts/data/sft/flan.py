import argparse

from datasets import concatenate_datasets, load_dataset

import open_instruct.utils as open_instruct_utils
from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Flan dataset and optionally upload to Hugging Face Hub.")
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
        default="flan_v2_converted",
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
        "This is a converted version of the Flan dataset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/flan.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "The original FLAN dataset needs extensive efforts to be regenerated, "
        "so we are using [a reproduced version by the OpenOrca team](https://huggingface.co/datasets/Open-Orca/FLAN)."
        "More specifically, we only use their top level jsonl files, which is a subset of the original dataset."
        "And by default, we only use the `cot_fsopt_data`, `cot_zsopt_data`, `niv2_fsopt_data`, `niv2_zsopt_data` "
        "`flan_fsopt_data`, `flan_zsopt_data`, `t0_fsopt_data` subsets."
        "If you want to use more data, you can modify this script to load more data from their Huggingface repo."
        "Please refer to their Huggingface repo [here](https://huggingface.co/datasets/Open-Orca/FLAN) "
        "and the [original FLAN v2 repo](https://github.com/google-research/FLAN/tree/main/flan/v2) "
        "for more information about this dataset and the license."
    )

    sampling_sizes = {
        "cot_fsopt_data": 20000,
        "cot_zsopt_data": 20000,
        "niv2_fsopt_data": 20000,
        "niv2_zsopt_data": 20000,
        "flan_fsopt_data": 2000,
        "flan_zsopt_data": 2000,
        "t0_fsopt_data": 6000,
    }

    subsets = []
    for subset, sampling_size in sampling_sizes.items():
        ds = load_dataset(
            "Open-Orca/FLAN", data_files=f"{subset}/*", num_proc=open_instruct_utils.max_num_processes()
        )["train"]
        if len(ds) > sampling_size:
            ds = ds.shuffle(seed=42).select(range(sampling_size))
        subsets.append(ds)

    ds = concatenate_datasets(subsets)

    conversion_func = lambda example: {
        "messages": [
            {"role": "user", "content": example["inputs"]},
            {"role": "assistant", "content": example["targets"]},
        ]
    }
    convert_sft_dataset(
        ds=ds,
        hf_dataset_id=None,
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name=args.converted_dataset_name,
        local_save_dir=args.local_save_dir,
        readme_content=readme_content,
    )
