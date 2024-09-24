

import json
from datasets import load_dataset
from tqdm import tqdm


def process_pku_saferlhf(output_file):

    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

    with open(output_file, "w") as outfile:
        for split in dataset.keys():
            for item in tqdm(dataset[split], desc=f"Processing {split}"):
                if item["safer_response_id"]  == 0:
                    chosen = item["response_0"]
                    rejected = item["response_1"]
                else:
                    chosen = item["response_1"]
                    rejected = item["response_0"]

            preference_data = {
                "chosen": chosen,
                "rejected": rejected
            }
            json.dump(preference_data, outfile)
            outfile.write("\n")

    print("Processing done!")


if __name__ == "__main__":
    output_file = "/net/nfs.cirrascale/mosaic/nouhad/projects/open-instruct/open-instruct/safety/pku_saferlhf_preferences.jsonl"
    process_pku_saferlhf(output_file)