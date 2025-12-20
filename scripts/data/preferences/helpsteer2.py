import argparse
from collections import defaultdict

import datasets

ALL_ASPECTS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]


def average_score(example, aspects_to_consider):
    return sum(example[aspect] for aspect in aspects_to_consider) / len(aspects_to_consider)


def binarize_dataset(dataset, aspects_to_ignore: list[str], min_score: float | None, margin: float | None):
    aspects_to_consider = [aspect for aspect in ALL_ASPECTS if aspect not in aspects_to_ignore]
    prompt_groups = defaultdict(list)

    for example in dataset:
        prompt_groups[example["prompt"]].append(example)

    binarized_data = []
    for prompt, responses in prompt_groups.items():
        if len(responses) < 2:
            continue  # Skip prompts with only one response

        sorted_responses = sorted(responses, key=lambda x: average_score(x, aspects_to_consider), reverse=True)
        chosen = sorted_responses[0]
        rejected = sorted_responses[-1]

        chosen_avg_score = average_score(chosen, aspects_to_consider)
        rejected_avg_score = average_score(rejected, aspects_to_consider)

        # Skip this datapoint if the scores are tied
        if chosen_avg_score == rejected_avg_score:
            continue

        # Skip this datapoint if the chosen score is below the minimum threshold
        if min_score is not None and chosen_avg_score < min_score:
            continue

        # Skip this datapoint if the margin between chosen and rejected is less than specified
        if margin is not None and (chosen_avg_score - rejected_avg_score) < margin:
            continue

        binarized_example = {
            "prompt": prompt,
            "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected["response"]}],
            "avg_scores": (round(chosen_avg_score, 2), round(rejected_avg_score, 2)),
        }
        binarized_data.append(binarized_example)

    return datasets.Dataset.from_list(binarized_data)


def main(
    aspects_to_ignore: list[str],
    min_score: float | None,
    margin: float | None,
    push_to_hub: bool,
    hf_entity: str | None,
):
    # Load the HelpSteer 2 dataset
    ds = datasets.load_dataset("nvidia/HelpSteer2", num_proc=max_num_processes())

    # Binarize the dataset
    binarized_ds = binarize_dataset(ds["train"], aspects_to_ignore, min_score, margin)

    dataset_name = "helpsteer-2-binarized"
    if min_score is not None:
        dataset_name += f"-above-{min_score}"
    if margin is not None:
        dataset_name += f"-margin-{margin}"
    if aspects_to_ignore:
        ignored_aspects_str = "-".join(sorted(aspects_to_ignore))
        dataset_name += f"-ignore-{ignored_aspects_str}"

    print(f"Creating binarized dataset: {dataset_name}")
    print(f"Original dataset size: {int(len(ds['train'])/2)}")
    print(f"Binarized dataset size: {len(binarized_ds)}")
    print(
        f"Removed {len(ds['train']) - len(binarized_ds)} datapoints due to ties, single responses, below threshold, or insufficient margin"
    )

    if push_to_hub:
        if hf_entity:
            full_name = f"{hf_entity}/{dataset_name}"
        else:
            full_name = dataset_name

        print(f"Pushing dataset to Hub: {full_name}")
        binarized_ds.push_to_hub(full_name, private=True)
    else:
        print(f"Dataset {dataset_name} created locally (not pushed to Hub)")

    return binarized_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Binarize HelpSteer 2 dataset and optionally upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--aspects_to_ignore", nargs="*", default=[], help="Aspects to ignore during binarization (default: none)"
    )
    parser.add_argument(
        "--min_score", type=float, default=None, help="Minimum average score for chosen responses (default: None)"
    )
    parser.add_argument(
        "--margin", type=float, default=None, help="Minimum margin between chosen and rejected scores (default: None)"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )

    args = parser.parse_args()
    # assert that the aspects to ignore are valid
    for aspect in args.aspects_to_ignore:
        if aspect not in ALL_ASPECTS:
            raise ValueError(f"Invalid aspect to ignore: {aspect}. Must be one of {ALL_ASPECTS}")
    main(args.aspects_to_ignore, args.min_score, args.margin, args.push_to_hub, args.hf_entity)
