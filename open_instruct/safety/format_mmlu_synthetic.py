from datasets import load_dataset, Dataset
import json
from typing import List

def load_and_format_data(dataset_name, split="train"):
    dataset = load_dataset(dataset_name, split=split)

    formatted_data = []

    for item in dataset:
        question = item['question']
        choices = item['choices']
        answer_index = item['answer']

        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = f"{question}\n\nChoices:\n{choices_text}"

        try:
            response = int(choices[answer_index])
        except ValueError:
            print(f"Warning: Could not convert answer to integer for question: {question}")
            response = choices[answer_index]

        conversation_block = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": f"The correct answer is: {response}"
                }
            ]
        }
        formatted_data.append(conversation_block)

    return formatted_data

def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "user_message": [item["messages"][0]["content"] for item in data],
            "assistant_message": [item["messages"][1]["content"] for item in data],
            "subject": [item["messages"][0]["content"].split('\n')[0] for item in data],
        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")

# Use the specific dataset
dataset_name = "ai2-adapt-dev/synthetic-mmlu-dataset"
formatted_data = load_and_format_data(dataset_name)

# Save the formatted data to a JSON file
with open('formatted_synthetic_mmlu_data.json', 'w') as f:
    json.dump(formatted_data, f, indent=2)

print(f"Formatted data saved to 'formatted_synthetic_mmlu_data.json'")

# Print a sample conversation block
print("\nSample conversation block:")
print(json.dumps(formatted_data[0], indent=2))

# Print some statistics
print(f"\nTotal number of questions: {len(formatted_data)}")
subjects = set(item['messages'][0]['content'].split('\n')[0] for item in formatted_data)
print(f"Number of unique subjects: {len(subjects)}")
print("Subjects:", ", ".join(sorted(subjects)))

# Check for any non-integer answers
non_integer_answers = [item for item in formatted_data if not isinstance(eval(item['messages'][1]['content'].split(': ')[1]), int)]
print(f"\nNumber of non-integer answers: {len(non_integer_answers)}")
if non_integer_answers:
    print("Sample non-integer answer:")
    print(json.dumps(non_integer_answers[0], indent=2))

# Upload the formatted data to Hugging Face
upload_to_huggingface(formatted_data, "ai2-adapt-dev/formatted_synthetic_mmlu_data")