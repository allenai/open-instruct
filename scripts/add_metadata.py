import json
import os

from huggingface_hub import HfApi

# metadata scheme:
# {
#   "model_name": "Meta-Llama-3.1-8B-Instruct",
#   "model_type": "sft",
#   "datasets": [],
#   "base_model": "meta-llama/Meta-Llama-3.1-8B",
#   "wandb_path": "https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct",
#   "beaker_datasets": []
# }


def take_and_check_valid_input(prompt, valid_values):
    while True:
        print(prompt, end=" ")
        user_input = input()
        if user_input in valid_values:
            return user_input
        else:
            print(f"Invalid input. Please enter one of {valid_values}.")


print("Welcome! This script will upload metadata for a run if you need. Please provide the following information.")
print("Please ensure you have set a write access HF token in the HF_TOKEN environment variable.")
print("Model name (e.g. Meta-Llama-3.1-8B-Instruct):", end=" ")
model_name = input()
valid_model_types = ["sft", "dpo", "ppo", "merge", "reject", "online-dpo"]
model_type = take_and_check_valid_input(f"Model type {valid_model_types}:", valid_model_types)
wandb_link = input("Link to wandb run: ")
beaker_datasets = [input("Beaker dataset: ")]
base_model = input(
    "Base model HF name (e.g. meta-llama/Meta-Llama-3.1-8B, or the starting SFT checkpoint for DPO/etc models): "
)
print("Thanks! Uploading metadata...")
# construct metadata
metadata = {
    "model_name": model_name,
    "model_type": model_type,
    "datasets": [],
    "base_model": base_model,
    "wandb_path": wandb_link,
    "beaker_datasets": beaker_datasets,
}

with open("metadata.json", "w") as f:
    json.dump(metadata, f)
for _ in range(3):
    try:
        api = HfApi(token=os.getenv("HF_TOKEN", None))
        api.upload_file(
            path_or_fileobj="metadata.json",
            path_in_repo=f"results/{model_name}/metadata.json",
            repo_id="allenai/tulu-3-evals",
            repo_type="dataset",
        )
        os.remove("metadata.json")
        print("Metadata uploaded successfully!")
        break
    except Exception as e:
        print(f"Error uploading metadata: {e}")
        print("Retrying...")
