from datasets import load_dataset

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

    return chosen_turn_counts, rejected_turn_counts

# Example usage
dataset_name = 'Anthropic/hh-rlhf'
chosen_turn_counts, rejected_turn_counts = count_turns_per_conversation(dataset_name)

# Print the turn counts for each conversation
for i, (chosen_count, rejected_count) in enumerate(zip(chosen_turn_counts, rejected_turn_counts), start=1):
    print(f'Conversation {i} has {chosen_count} turns in "chosen" and {rejected_count} turns in "rejected".')
