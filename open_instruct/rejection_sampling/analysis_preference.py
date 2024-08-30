from datasets import load_dataset


def count_turns_by_source(dataset):
    # Initialize counters
    turn_counts = {}

    # Process each split in the dataset
    for split in dataset.keys():
        split_data = dataset[split]
        turn_counts[split] = {"user_avg_turn_ch": [], "assistant_avg_turn_ch": [], "user_avg_turn_rej": [],
                              "assistant_avg_turn_rej": []}

        # Process each example in the split
        for example in split_data:
            source = example.get("source")
            chosen = example.get("chosen", [])
            rejected = example.get("rejected", [])

            # Count turns in chosen
            turn_counts_user = 0
            turn_counts_ass = 0
            for conversation in chosen:
                role = conversation.get("role", [])
                if role == "user":
                    turn_counts_user += 1
                elif role == "assistant":
                    turn_counts_ass += 1

            turn_counts[split]["user_avg_turn_ch"].append(turn_counts_user)
            turn_counts[split]["assistant_avg_turn_ch"].append(turn_counts_ass)

            # Count turns in rejected
            turn_counts_user_rej = 0
            turn_counts_ass_rej = 0
            for conversation in rejected:
                role = conversation.get("role")
                if role == "user":
                    turn_counts_user_rej += 1

                elif role == "assistant":
                    turn_counts_ass_rej += 1

            turn_counts[split]["user_avg_turn_rej"].append(turn_counts_user_rej)
            turn_counts[split]["assistant_avg_turn_rej"].append(turn_counts_ass_rej)
    return turn_counts


# Load the dataset from HF
dataset = load_dataset("allenai/tulu-2.5-preference-data")

# Count turns by source
turn_counts = count_turns_by_source(dataset)

# Print the results
for split, counts in turn_counts.items():
    print(f"Dataset Split: {split}")
    print(f"  User turns: {sum(counts['user_avg_turn_ch'])/ len(counts['user_avg_turn_ch'])}")
    print(f"  Assistant turns: {sum(counts['assistant_avg_turn_ch'])/ len(counts['assistant_avg_turn_ch'])}")