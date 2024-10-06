import json
import re
from typing import List
from datasets import load_dataset, Dataset


def extract_final_response(content: str) -> str:
    match = re.search(r'Final Response:(.*?)(?=\n\n|$)', content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return content  # Return the entire content if "Final Response:" is not found


def process_item(item: dict) -> dict:
    messages = item['messages']
    user_message, final_response = None, None
    for msg in messages:
        if msg['role'] == 'user':
            user_message = msg["content"]
        if msg['role'] == 'assistant':
            final_response = extract_final_response(msg["content"])

    if user_message is None or not final_response:
        print(
            f"Warning: Skipping item. User message: {'Present' if user_message else 'Missing'}, Final response: {'Present' if final_response else 'Missing'}")
        return None

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_response}
        ],
        "prompt_harm_label": item["prompt_harm_label"],
        "model_used": item["model_used"]
    }


def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "messages": [item["messages"] for item in data],
            "prompt_harm_label": [item["prompt_harm_label"] for item in data],
            "model_used": [item["model_used"] for item in data],
        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")
    print(f"Each row in the dataset represents a separate conversation between a user and the assistant.")


def main():
    dataset = load_dataset("ai2-adapt-dev/synthetic-cot-wildguarmixtrain")
    processed_data = [item for item in (process_item(item) for item in dataset['train']) if item is not None]
    print(f"Processed {len(processed_data)} items out of {len(dataset['train'])} total items.")
    upload_to_huggingface(processed_data, "ai2-adapt-dev/synthetic-finalresp-wildguarmixtrain")


if __name__ == "__main__":
    main()