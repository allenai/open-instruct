import argparse
import multiprocessing

import datasets
from huggingface_hub import HfApi


def is_not_tie(item):
    return item["score_0"] != item["score_1"]


def extract(item):
    # System prompt
    system_prompt = {
        "role": "system",
        "content": "You are a helpful assistant answering questions. You will get a question with some excerpts from websites that could be useful to answer them.",
    }

    def format_quotes(quotes):
        if isinstance(quotes, str):
            return quotes
        elif isinstance(quotes, dict):
            extracts = quotes.get("extract", [])
            if isinstance(extracts, list):
                return "\n\n".join(f"{i+1}. {extract}" for i, extract in enumerate(extracts))
            else:
                return extracts
        elif isinstance(quotes, list):
            return "\n\n".join(f"{i+1}. {extract}" for i, extract in enumerate(quotes))
        else:
            return ""

    # Prepare the numbered quotes for each answer
    quotes_0 = format_quotes(item.get("quotes_0", ""))
    quotes_1 = format_quotes(item.get("quotes_1", ""))

    # Prepare the user messages (questions with quotes)
    question_0 = {"role": "user", "content": f"{item['question']['full_text']}\n\n{quotes_0}"}
    question_1 = {"role": "user", "content": f"{item['question']['full_text']}\n\n{quotes_1}"}

    # Prepare the assistant messages (answers)
    answer_0 = {"role": "assistant", "content": item["answer_0"]}
    answer_1 = {"role": "assistant", "content": item["answer_1"]}

    # Create the full messages
    message_0 = [system_prompt, question_0, answer_0]
    message_1 = [system_prompt, question_1, answer_1]

    # Prepare the example dictionary
    example = {"prompt": item["question"]["full_text"]}

    # Compare scores and assign chosen/rejected
    if item["score_0"] > item["score_1"]:
        example["chosen"] = message_0
        example["rejected"] = message_1
    elif item["score_1"] > item["score_0"]:
        example["chosen"] = message_1
        example["rejected"] = message_0

    # add margin score that is the difference between the two scores
    example["margin"] = abs(item["score_0"] - item["score_1"])

    return example


def main(push_to_hub: bool, hf_entity: str | None):
    api = HfApi()

    ds = datasets.load_dataset("openai/webgpt_comparisons", num_proc=max_num_processes())

    # filter out the ties
    ds = ds["train"].filter(is_not_tie, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

    # test one element
    print(extract(ds[0]))

    ds = ds.map(extract, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)

    all_col = ds.column_names
    # remove except prompt and chosen and rejected
    ds = ds.remove_columns([col for col in all_col if col not in ["prompt", "chosen", "rejected", "margin"]])

    print(f"{multiprocessing.cpu_count()=}")

    if push_to_hub:
        if hf_entity:
            repo_id = f"{hf_entity}/webgpt-binarized"
        else:
            repo_id = "webgpt-binarized"
        print(f"Pushing dataset to Hub: {repo_id}")
        ds.push_to_hub(repo_id)

        api.upload_file(
            path_or_fileobj=__file__, path_in_repo="create_dataset.py", repo_id=repo_id, repo_type="dataset"
        )
    else:
        print("Dataset processed locally (not pushed to Hub)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process WebGPT dataset and optionally upload to Hugging Face Hub.")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )

    args = parser.parse_args()

    main(args.push_to_hub, args.hf_entity)
