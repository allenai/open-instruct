import json
import re
from typing import List
from datasets import load_dataset, Dataset


def extract_final_response(content: str) -> str:
    match = re.search(r'Final Response:\s*(.*?)(?:\n\n|$)', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def process_item(item: dict) -> dict:
    # Each item represents a separate conversation
    messages = json.loads(item['messages'])
    user_message = next(msg for msg in messages if msg['role'] == 'user')
    assistant_message = next(msg for msg in messages if msg['role'] == 'assistant')

    final_response = extract_final_response(assistant_message['content'])

    return {
        "messages": [  # Renamed from "messages" to "conversation" for clarity
            {"role": "user", "content": user_message['content']},
            {"role": "assistant", "content": final_response}
        ],
        "prompt_harm_label": item["prompt_harm_label"],
        "model_used": item["model_used"]
    }


def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "conversation": [item["conversation"] for item in data],  # Each item is a separate conversation
            "prompt_harm_label": [item["prompt_harm_label"] for item in data],
            "model_used": [item["model_used"] for item in data],
        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")
    print(f"Each row in the dataset represents a separate conversation between a user and the assistant.")


def main():
    # Load the dataset
    dataset = load_dataset("ai2-adapt-dev/synthetic-cot-wildguarmixtrain")

    # Process the data
    processed_data = [process_item(item) for item in dataset['train']]

    # Upload to Hugging Face
    upload_to_huggingface(processed_data, "ai2-adapt-dev/synthetic-finalresp-wildguarmixtrain")


if __name__ == "__main__":
    main()