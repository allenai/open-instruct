import re
import uuid

from datasets import load_dataset

import open_instruct.utils as open_instruct_utils

LLAMA_NEMOTRON_REPO = "nvidia/Llama-Nemotron-Post-Training-Dataset"
SFT_DEST_REPO = "saurabh5/llama-nemotron-sft"


def extract_python_code(model_output: str) -> str:
    """Extract the last code block between ``` markers from the model output."""
    # Find content between ``` markers
    pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(pattern, model_output, re.DOTALL)

    if not matches:
        return "NO SOLUTION"

    # Return the first match, stripped of whitespace
    return matches[-1].strip()


def process_and_upload_dataset():
    print(f"Loading dataset {LLAMA_NEMOTRON_REPO}, subset SFT...")
    dataset = load_dataset(LLAMA_NEMOTRON_REPO, "SFT", split="code", num_proc=open_instruct_utils.max_num_processes())
    print(f"Dataset loaded with {len(dataset)} examples, proceeed?")
    breakpoint()

    def transform_row(example):
        inputs_list = example["input"]
        new_example = example.copy()
        messages = list(inputs_list)
        messages.append({"role": "assistant", "content": example["output"]})
        new_example["messages"] = messages
        new_example["solution"] = extract_python_code(example["output"])
        new_example["id"] = str(uuid.uuid4())
        return new_example

    print("Transforming data to create 'messages' column...")
    transformed_dataset = dataset.map(transform_row, num_proc=4)

    print("Renaming 'category' column to 'dataset'...")
    if "category" in transformed_dataset.column_names:
        transformed_dataset = transformed_dataset.rename_column("category", "dataset")
    else:
        print("Warning: 'category' column not found, skipping rename.")

    print(f"Uploading processed dataset to {SFT_DEST_REPO}...")
    transformed_dataset.push_to_hub(SFT_DEST_REPO, private=False)

    print("Dataset processing and uploading complete.")


if __name__ == "__main__":
    process_and_upload_dataset()
