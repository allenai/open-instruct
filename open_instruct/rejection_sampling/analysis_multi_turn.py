from datasets import load_dataset

def count_turns_per_conversation(dataset_name):
    # Load the dataset
    dataset = load_dataset(dataset_name, split='train')  # Assuming we are using the 'train' split

    turn_counts = []

    # Iterate through each conversation in the dataset
    for conversation in dataset:
        # Extract the conversation text
        conversation_text = conversation['text']  # Adjust this key if necessary

        # Split the conversation text into lines
        lines = conversation_text.strip().split('\n')

        # Count the number of turns by counting how many times "Human:" or "Assistant:" appears
        turn_count = sum(1 for line in lines if line.startswith('Human:') or line.startswith('Assistant:'))

        # Append the turn count to the list
        turn_counts.append(turn_count)

    return turn_counts

# Example usage
dataset_name = 'Anthropic/hh-rlhf'
turn_counts = count_turns_per_conversation(dataset_name)

print("Done")
# Print the turn counts for each conversation
for i, count in enumerate(turn_counts, start=1):
    print(f'Conversation {i} has {count} turns.')
