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
        subject = item['subject']  # Reading subject from the data item

        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        prompt = f"{question}\n\nChoices:\n{choices_text}"

        try:
            response = choices[int(answer_index)]
        except ValueError:
            print(f"Warning: Could not convert answer index to integer for question: {question}")
            response = choices[answer_index]

        conversation_block = {
            "subject": subject,  # Adding subject to the conversation block
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
            "subject": [item["subject"] for item in data],
            "messages": [item["messages"] for item in data],
            "model_used": ["gpt-4" for _ in data],  # Adding "model used" column
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

# Upload the formatted data to Hugging Face
upload_to_huggingface(formatted_data, "ai2-adapt-dev/formatted_synthetic_mmlu_data")