import json
from datasets import load_dataset, Dataset
from typing import List, Dict, Optional
from tqdm import tqdm


def upload_to_huggingface(data: List[dict], dataset_name: str):
    dataset = Dataset.from_dict(
        {
            "chosen": [item["chosen"] for item in data],
            "rejected": [item["rejected"] for item in data],
            "source": [item["source"] for item in data],

        }
    )

    dataset.push_to_hub(dataset_name)
    print(f"Dataset uploaded to Hugging Face: https://huggingface.co/datasets/{dataset_name}")

def process_pku_saferlhf(output_file):
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

    with open(output_file, "w") as outfile:
        preference  = []
        for split in dataset.keys():
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                if item["safer_response_id"] == '0':  # Changed to string '0'
                    chosen = item["response_0"]
                    rejected = item["response_1"]
                else:
                    chosen = item["response_1"]
                    rejected = item["response_0"]

                prompt = item["prompt"]

                preference_data = {
                    "chosen": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": chosen}
                    ],
                    "rejected": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": rejected}
                    ],
                    "source": "PKU-SafeRLHF"
                }
                json.dump(preference_data, outfile)
                preference.append(preference_data)
                outfile.write('\n')  # Add newline for JSONL format

    print("Processing done!")
    print(f"Data saved in {output_file}")

    upload_to_huggingface(preference, "ai2-adapt-dev/PKU-SafeRLHF-processed")


if __name__ == "__main__":
    output_file = "/net/nfs.cirrascale/mosaic/nouhad/projects/open-instruct/open_instruct/safety/pku_saferlhf_preferences.jsonl"
    process_pku_saferlhf(output_file)