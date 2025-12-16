import argparse
import random

from datasets import Dataset, load_dataset
from tqdm import tqdm

import open_instruct.utils as open_instruct_utils


def clean_prompt(prompt):
    if prompt.startswith("Human:"):
        prompt = prompt[6:]
    if prompt.endswith("Assistant:"):
        prompt = prompt[:-10]
    return prompt.strip()


def is_valid_response(response):
    return "Human:" not in response and "Assistant:" not in response


def load_nectar_dataset(subset="lmsys-chat-1m", deduplication=False):
    # Load the Nectar dataset
    nectar_dataset = load_dataset(
        "berkeley-nest/Nectar", split="train", num_proc=open_instruct_utils.max_num_processes()
    )
    print(f"Original Nectar dataset size: {len(nectar_dataset)}")

    if deduplication:
        # We must deduplicate the dataset from UltraFeedback given both the datasets are
        # very popular and sourcing many of the same prompts (FLAN, ShareGPT, evol instruct, etc)
        # we handle LMSYS and Anthropic HH separately because UltraFeedback does not use these
        print("Deduplication enabled. Loading UltraFeedback dataset...")
        ultra_feedback = load_dataset(
            "openbmb/UltraFeedback", split="train", num_proc=open_instruct_utils.max_num_processes()
        )
        print(f"UltraFeedback dataset size: {len(ultra_feedback)}")

        # Create a set of UltraFeedback instructions for faster lookup
        ultra_feedback_instructions = set(ultra_feedback["instruction"])

        # Filter Nectar dataset, excluding lmsys-chat-1m and anthropic-hh
        filtered_dataset = nectar_dataset.filter(
            lambda example: example["source"][0] not in ["lmsys-chat-1m", "anthropic-hh"]
        )
    else:
        # Filter the dataset based on the subset
        filtered_dataset = nectar_dataset.filter(lambda example: example["source"] == [subset])

    print(f"Filtered dataset size: {len(filtered_dataset)}")

    binarized_data = []
    excluded_count = 0

    # Use tqdm for progress bar
    for example in tqdm(filtered_dataset, desc="Processing examples", unit="example"):
        prompt = clean_prompt(example["prompt"])

        # Skip if the prompt is in UltraFeedback dataset (when deduplication is enabled)
        if deduplication and prompt in ultra_feedback_instructions:
            # print found duplicate
            print(f"Found duplicate prompt: {prompt}")
            excluded_count += 1
            continue

        answers = example["answers"]
        sorted_answers = sorted(answers, key=lambda x: x["rank"])
        valid_answers = [a for a in sorted_answers if is_valid_response(a["answer"])]

        if len(valid_answers) >= 2:
            top_n = min(4, len(valid_answers))
            chosen = random.choice(valid_answers[:top_n])
            remaining_answers = [a for a in valid_answers if a != chosen]
            rejected = random.choice(remaining_answers)

            binarized_example = {
                "prompt": prompt,
                "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen["answer"]}],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected["answer"]},
                ],
            }

            binarized_data.append(binarized_example)
        else:
            excluded_count += 1

    print(f"Number of binarized examples: {len(binarized_data)}")
    print(f"Number of excluded examples: {excluded_count}")

    return binarized_data


def main():
    parser = argparse.ArgumentParser(description="Load and process the Nectar dataset")
    parser.add_argument("--dataset_name", type=str, default="nectar_binarized", help="Name for the processed dataset")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to Hugging Face Hub")
    parser.add_argument("--hf_entity", type=str, default=None, help="Hugging Face username or organization")
    parser.add_argument(
        "--deduplication", action="store_true", help="Apply deduplication to the dataset from UltraFeedback"
    )
    parser.add_argument("--subset", type=str, default="lmsys-chat-1m", help="Subset of the dataset to use")

    args = parser.parse_args()

    # Load and process the dataset
    binarized_data = load_nectar_dataset(subset=args.subset, deduplication=args.deduplication)

    # Convert to Hugging Face Dataset
    binarized_ds = Dataset.from_list(binarized_data)

    # Dataset naming logic
    dataset_name = args.dataset_name
    if args.deduplication:
        dataset_name += "-dedup-ultrafeedback"
    else:
        dataset_name += f"-{args.subset}"

    # Saving logic
    if args.push_to_hub:
        if args.hf_entity:
            full_name = f"{args.hf_entity}/{dataset_name}"
        else:
            full_name = dataset_name

        print(f"Pushing dataset to Hub: {full_name}")
        binarized_ds.push_to_hub(full_name, private=True)
    else:
        print(f"Dataset {dataset_name} created locally (not pushed to Hub)")
        # Optionally, you can save the dataset locally here
        # binarized_ds.save_to_disk(dataset_name)

    print("First example:")
    print(binarized_data[0])


if __name__ == "__main__":
    main()
