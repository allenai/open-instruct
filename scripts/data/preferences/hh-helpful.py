import argparse
import multiprocessing

import datasets
from huggingface_hub import HfApi


def find_all_occurrences(substring, string):
    """Find all occurrences of a substring in a string and return their indices."""
    indices = []
    index = string.find(substring)
    while index != -1:
        indices.append(index)
        index = string.find(substring, index + 1)
    return indices


def extract(example):
    result = {}
    for key in ["chosen", "rejected"]:
        human_indices = find_all_occurrences("\n\nHuman: ", example[key])
        assistant_indices = find_all_occurrences("\n\nAssistant: ", example[key])
        combined_indices = []
        for human_index, assistant_index in zip(human_indices, assistant_indices):
            combined_indices.append(human_index)
            combined_indices.append(assistant_index)
        combined_indices.append(len(example[key]))
        structured_data = []
        for i in range(len(combined_indices) - 1):
            role_flag = "user" if i % 2 == 0 else "assistant"
            content_offset_idx = len("\n\nHuman: ") if i % 2 == 0 else len("\n\nAssistant: ")
            content = example[key][combined_indices[i] + content_offset_idx : combined_indices[i + 1]]
            structured_data.append({"role": role_flag, "content": content})
        result[key] = structured_data
    result["messages"] = result["chosen"]
    return result


def not_contains_empty(data):
    for msg in data["chosen"]:
        if not msg["content"]:
            return False
    for msg in data["rejected"]:
        if not msg["content"]:
            return False
    return True


def ends_with_assistant(data):
    return data["chosen"][-1]["role"] == "assistant" and data["rejected"][-1]["role"] == "assistant"


def filter_fn(data):
    return not_contains_empty(data) and ends_with_assistant(data)


def main(push_to_hub: bool, hf_entity: str | None):
    api = HfApi()

    ds = datasets.load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", num_proc=max_num_processes())

    ds = ds.map(extract, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

    ds = ds.filter(filter_fn, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

    print(f"{multiprocessing.cpu_count()=}")

    if push_to_hub:
        if hf_entity:
            repo_id = f"{hf_entity}/hh-rlhf-helpful"
        else:
            repo_id = "hh-rlhf-helpful"

        print(f"Pushing dataset to Hub: {repo_id}")
        ds.push_to_hub(repo_id)

        api.upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_id=repo_id, repo_type="dataset"
        )
    else:
        print("Dataset processed locally (not pushed to Hub)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process HH-RLHF helpful dataset and optionally upload to Hugging Face Hub."
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
