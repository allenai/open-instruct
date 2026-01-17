import argparse

import datasets


def main(push_to_hub: bool, hf_entity: str | None):
    ds = datasets.load_dataset("allenai/tulu-2.5-preference-data", num_proc=max_num_processes())

    # for each config in the above dict, upload a new private dataset
    # with the corresponding subsets, for easy mixing

    for split in ds.keys():
        dataset_name = f"tulu-2.5-prefs-{split}"
        print(f"Creating dataset {dataset_name}...")

        if push_to_hub:
            if hf_entity:
                full_name = f"{hf_entity}/{dataset_name}"
            else:
                full_name = dataset_name

            print(f"Pushing dataset to Hub: {full_name}")
            ds[split].push_to_hub(full_name, private=True)
        else:
            print(f"Dataset {dataset_name} created locally (not pushed to Hub)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Tulu 2.5 preference data and optionally upload to Hugging Face Hub."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )

    args = parser.parse_args()

    main(args.push_to_hub, args.hf_entity)
