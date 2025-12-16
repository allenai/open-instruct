import argparse

from datasets import Dataset, load_dataset

import open_instruct.utils as open_instruct_utils
from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Tulu hard-coded examples and optionally upload to Hugging Face Hub."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )
    parser.add_argument(
        "--converted_dataset_name", type=str, default=None, help="Name of the converted dataset on Hugging Face Hub"
    )
    parser.add_argument("--local_save_dir", type=str, default=None, help="Local directory to save the dataset")
    parser.add_argument("--repeat_n", type=int, default=1, help="Number of times to repeat the hard-coded examples")
    args = parser.parse_args()

    ds = load_dataset("allenai/tulu-3-hardcoded-prompts", num_proc=open_instruct_utils.max_num_processes())["train"]

    readme_content = (
        "This dataset contains a set of hard-coded examples for Tulu, "
        "a virtual assistant developed by AI2. "
        "These examples are designed to provide the identity and some basic information about Tulu and its creators.\n\n"
        f"The original hard-coded set contains {len(ds)} examples, "
        f"and we repeat each example {args.repeat_n} times to create this training dataset.\n"
        "The creation script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/tulu_hard_coded.py) repo.\n"
    )

    ds = Dataset.from_list(ds.to_list() * args.repeat_n)

    if args.converted_dataset_name is None:
        args.converted_dataset_name = f"tulu_hard_coded_repeated_{args.repeat_n}"

    convert_sft_dataset(
        ds=ds,
        hf_dataset_id=None,
        convert_fn=None,
        apply_keyword_filters=False,
        apply_empty_message_filters=False,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name=args.converted_dataset_name,
        local_save_dir=args.local_save_dir,
        readme_content=readme_content,
    )
