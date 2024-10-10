from datasets import load_dataset, Dataset
from typing import List, Dict


def process_item(item: Dict) -> Dict:
    user_message = item['vanilla'] if item['vanilla'] else item['adversarial']
    assistant_message = item['completion']

    if not user_message or not assistant_message:
        return None

    return {
        "messages": [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ],
        "prompt_harm_label": item["data_type"],
        "model_used": item.get("model", "unknown")  # Assuming the model info is available, otherwise "unknown"
    }


def upload_to_huggingface(data: List[Dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "messages": [item["messages"] for item in data],
            "prompt_harm_label": [item["prompt_harm_label"] for litem in data],
            "model_used": [item["model_used"] for item in data],
        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")
    print(f"Each row in the dataset represents a separate conversation between a user and the assistant.")


def main():
    # Load the dataset
    dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)

    # Process the data
    processed_data = [
        item for item in (process_item(row) for row in dataset['train'])
        if item is not None
    ]

    print(f"Processed {len(processed_data)} items out of {len(dataset['train'])} total items.")

    # Upload to Hugging Face
    upload_to_huggingface(processed_data, "ai2-adapt-dev/processed-wildjailbreak")


if __name__ == "__main__":
    main()