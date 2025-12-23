import argparse
from collections import defaultdict

import datasets

ALL_ASPECTS = ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]

WEIGHT_SETS = [
    # 340b
    {"helpfulness": 0.3, "correctness": 0.74, "coherence": 0.46, "complexity": 0.47, "verbosity": -0.33},
    # 70b helpsteer2 lm
    {"helpfulness": 0.65, "correctness": 0.8, "coherence": 0.45, "complexity": 0.55, "verbosity": -0.4},
]


def weighted_score(example, weights):
    return sum(weights[aspect] * example[aspect] for aspect in ALL_ASPECTS)


def binarize_dataset(dataset, min_score: float | None):
    prompt_groups = defaultdict(list)

    for example in dataset:
        prompt_groups[example["prompt"]].append(example)

    binarized_data = []
    for prompt, responses in prompt_groups.items():
        if len(responses) < 2:
            continue  # Skip prompts with only one response

        # weights = random.choice(WEIGHT_SETS)
        weights = WEIGHT_SETS[1]  # change this to 0 for 340B config

        sorted_responses = sorted(responses, key=lambda x: weighted_score(x, weights), reverse=True)
        chosen = sorted_responses[0]
        rejected = sorted_responses[-1]

        chosen_score = weighted_score(chosen, weights)
        rejected_score = weighted_score(rejected, weights)

        # Skip this datapoint if the scores are tied
        if chosen_score == rejected_score:
            continue

        # Skip this datapoint if the chosen score is below the minimum threshold
        if min_score is not None and chosen_score < min_score:
            continue

        binarized_example = {
            "prompt": prompt,
            "chosen": [{"role": "user", "content": prompt}, {"role": "assistant", "content": chosen["response"]}],
            "rejected": [{"role": "user", "content": prompt}, {"role": "assistant", "content": rejected["response"]}],
            "weighted_scores": (round(chosen_score, 2), round(rejected_score, 2)),
        }
        binarized_data.append(binarized_example)

    return datasets.Dataset.from_list(binarized_data)


def main(min_score: float | None, push_to_hub: bool, hf_entity: str | None):
    # Load the HelpSteer 2 dataset
    ds = datasets.load_dataset("nvidia/HelpSteer2", num_proc=max_num_processes())

    # Binarize the dataset
    binarized_ds = binarize_dataset(ds["train"], min_score)

    dataset_name = "helpsteer2-binarized-nvidia-spec"
    if min_score is not None:
        dataset_name += f"-above-{min_score}"

    print(f"Creating binarized dataset: {dataset_name}")
    print(f"Original dataset size: {int(len(ds['train'])/2)}")
    print(f"Binarized dataset size: {len(binarized_ds)}")
    print(
        f"Removed {len(ds['train']) - len(binarized_ds)} datapoints due to ties, single responses, or below threshold"
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
        description="Binarize HelpSteer 2 dataset using NVIDIA's specification and optionally upload to Hugging Face Hub."
    )
    parser.add_argument(
        "--min_score", type=float, default=None, help="Minimum weighted score for chosen responses (default: None)"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument(
        "--hf_entity",
        type=str,
        default=None,
        help="Hugging Face organization to upload to (if not provided, uploads to user's account)",
    )

    args = parser.parse_args()
    main(args.min_score, args.push_to_hub, args.hf_entity)
