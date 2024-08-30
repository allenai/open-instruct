from datasets import load_dataset


def count_turns_by_source(dataset):
    # Initialize counters
    turn_counts = {}

    # Process each split in the dataset
    for split in dataset.keys():
        split_data = dataset[split]

        # Process each example in the split
        for example in split_data:
            messages = example.get("messages", [])
            dataset_name = example.get("dataset", "unknown")  # Provide a default value if dataset_name is missing

            # Initialize counters for this dataset
            if dataset_name not in turn_counts:
                turn_counts[dataset_name] = {"user_avg_turn_ch": [], "assistant_avg_turn_ch": []}

            # Count turns in conversation
            turn_counts_user = 0
            turn_counts_ass = 0
            for conversation in messages:
                role = conversation.get("role", "")
                if role == "user":
                    turn_counts_user += 1
                elif role == "assistant":
                    turn_counts_ass += 1

            # Append turn counts to the dataset's lists
            turn_counts[dataset_name]["user_avg_turn_ch"].append(turn_counts_user)
            turn_counts[dataset_name]["assistant_avg_turn_ch"].append(turn_counts_ass)

    return turn_counts


# Load the dataset from HF
dataset = load_dataset("allenai/tulu-v2-sft-mixture")

# Count turns by source
turn_counts = count_turns_by_source(dataset)

# Print the results
for split, counts in turn_counts.items():
    print(f"Dataset Split: {split}")
    print(f"  User turns: {sum(counts['user_avg_turn_ch'])/ len(counts['user_avg_turn_ch'])}")
    print(f"  Assistant turns: {sum(counts['assistant_avg_turn_ch'])/ len(counts['assistant_avg_turn_ch'])}")