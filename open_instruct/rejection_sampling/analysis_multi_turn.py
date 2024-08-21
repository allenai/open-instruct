from datasets import load_dataset
from collections import Counter

def count_turns_per_conversation(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, split='train')  # Assuming we are using the 'train' split

    chosen_turn_counts = []
    rejected_turn_counts = []

    # Iterate through each conversation in the dataset
    for conversation in dataset:
        # Extract the chosen and rejected conversation texts
        chosen_text = conversation['chosen']
        rejected_text = conversation['rejected']

        # Function to count turns in a conversation
        def count_turns(text):
            lines = text.strip().split('\n')
            return sum(1 for line in lines if line.startswith('Human:') or line.startswith('Assistant:'))

        # Count turns for both 'chosen' and 'rejected'
        chosen_turn_count = count_turns(chosen_text)
        rejected_turn_count = count_turns(rejected_text)

        # Append the turn counts to the respective lists
        chosen_turn_counts.append(chosen_turn_count)
        rejected_turn_counts.append(rejected_turn_count)

    # Use Counter to count how many conversations have each number of turns
    chosen_turn_distribution = Counter(chosen_turn_counts)
    rejected_turn_distribution = Counter(rejected_turn_counts)

    return chosen_turn_distribution, rejected_turn_distribution

# Example usage
dataset_name = 'Anthropic/hh-rlhf'
chosen_turn_distribution, rejected_turn_distribution = count_turns_per_conversation(dataset_name)

# Print the distributions
print("Chosen conversation turn distribution:")
for k, v in chosen_turn_distribution.items():
    print(f"{v} conversations have {k} turns.")

print("\nRejected conversation turn distribution:")
for k, v in rejected_turn_distribution.items():
    print(f"{v} conversations have {k} turns.")
