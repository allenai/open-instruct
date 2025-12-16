import argparse

import datasets
import numpy as np


def main(push_to_hub: bool, hf_entity: str | None):
    ex_url = "https://huggingface.co/datasets/ai2-adapt-dev/ultrafeedback-pipeline-replication/tree/main/setup_0"
    load_repo = "ai2-adapt-dev/ultrafeedback-pipeline-replication"
    dataset_name_base = "ultrafeedback-replication-p"
    setups = np.arange(7)

    for key in setups:
        json_path = f"setup_{key}"
        data_file = json_path + "/" + f"uf_setup_{key}_dpo_9000.jsonl"
        dataset = datasets.load_dataset(load_repo, data_files=data_file, num_proc=max_num_processes())

        dataset_name = f"{dataset_name_base}{key}"
        if push_to_hub:
            if hf_entity:
                full_name = f"{hf_entity}/{dataset_name}"
            else:
                full_name = dataset_name

            print(f"Pushing dataset to Hub: {full_name}")
            dataset.push_to_hub(full_name, private=True)
        else:
            print(f"Dataset {dataset_name} created locally (not pushed to Hub)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process UltraFeedback replication preference data and optionally upload to Hugging Face Hub."
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
