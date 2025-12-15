import argparse

from scripts.data.sft.utils import convert_sft_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process MetaMathQA dataset and optionally upload to Hugging Face Hub."
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

    def conversion_func(example):
        # the original dataset already has the `messages` key
        # our code here is mainly for keyword filtering
        return example

    cot_subset_readme_content = (
        "This is a converted version of the NuminaMath-CoT subset into Tulu SFT training format.\n\n"
        "The conversion script can be found in our "
        "[open-instruct](https://github.com/allenai/open-instruct/blob/main/scripts/data/sft/numinamath.py) repo.\n"
        "The conversion took the following parameters:\n"
        f"- apply_keyword_filters: {args.apply_keyword_filters}\n"
        f"- apply_empty_message_filters: {args.apply_empty_message_filters}\n"
        f"- push_to_hub: {args.push_to_hub}\n"
        f"- hf_entity: {args.hf_entity}\n"
        f"- converted_dataset_name: {args.converted_dataset_name}\n"
        f"- local_save_dir: {args.local_save_dir}\n\n"
        "Please refer to the [original CoT dataset](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) "
        "for more information about this dataset and the license."
    )

    convert_sft_dataset(
        ds=None,
        hf_dataset_id="AI-MO/NuminaMath-CoT",
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="numinamath_cot_converted"
        if args.converted_dataset_name is None
        else args.converted_dataset_name + "_cot",
        local_save_dir=args.local_save_dir,
        readme_content=cot_subset_readme_content,
    )

    tir_subset_readme_content = cot_subset_readme_content.replace("CoT", "TIR")

    convert_sft_dataset(
        ds=None,
        hf_dataset_id="AI-MO/NuminaMath-TIR",
        convert_fn=conversion_func,
        apply_keyword_filters=args.apply_keyword_filters,
        apply_empty_message_filters=args.apply_empty_message_filters,
        push_to_hub=args.push_to_hub,
        hf_entity=args.hf_entity,
        converted_dataset_name="numinamath_tir_converted"
        if args.converted_dataset_name is None
        else args.converted_dataset_name + "_tir",
        local_save_dir=args.local_save_dir,
        readme_content=tir_subset_readme_content,
    )
