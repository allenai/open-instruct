import random

from datasets import Dataset, load_dataset
from tqdm import tqdm

import open_instruct.utils as open_instruct_utils

NVIDIA_OCR = "nvidia/OpenCodeReasoning"
MY_OCR = "saurabh5/open-code-reasoning-rlvr"
SFT_REPO = "saurabh5/open-code-reasoning-sft"


def main():
    ocr_split0 = load_dataset(NVIDIA_OCR, "split_0", split="split_0", num_proc=open_instruct_utils.max_num_processes())
    my_ocr = load_dataset(MY_OCR, split="train", num_proc=open_instruct_utils.max_num_processes())

    id_to_data = {}
    for row in ocr_split0:
        datapoints_with_id = id_to_data.get(row["id"], [])
        datapoints_with_id.append(row)
        id_to_data[row["id"]] = datapoints_with_id

    # for every single data point in my_ocr, find all datapoints with the same "id" and upload them to the sft repo
    sft_data = []
    for row in tqdm(my_ocr, desc="Processing RLVR data"):
        datapoints_with_id = id_to_data.get(row["id"], [])
        for og_row in datapoints_with_id:
            sft_row = {
                **og_row,
                "messages": [
                    {"role": "user", "content": og_row["input"]},
                    {"role": "assistant", "content": og_row["output"]},
                ],
                "num_outputs": len(datapoints_with_id),
            }
            sft_data.append(sft_row)

    # upload the sft data to the sft repo
    dataset = Dataset.from_list(sft_data)
    dataset.push_to_hub(SFT_REPO)


def create_sft_dataset_with_n_outputs(n_outputs: list[int]):
    sft_base = load_dataset(SFT_REPO, split="train", num_proc=open_instruct_utils.max_num_processes())
    # for each id, sample exactly n_outputs datapoints
    id_to_data = {}
    for row in tqdm(sft_base, desc="Mapping id's in SFT data"):
        datapoints_with_id = id_to_data.get(row["id"], [])
        datapoints_with_id.append(row)
        id_to_data[row["id"]] = datapoints_with_id

    for n_output in tqdm(n_outputs, desc="Creating SFT datasets"):
        results = []
        for id_, datapoints_with_id in id_to_data.items():
            if len(datapoints_with_id) >= n_output:
                results.extend(random.sample(datapoints_with_id, k=n_output))

        dataset_name = f"{SFT_REPO}-n-{n_output}"
        dataset = Dataset.from_list(results)
        print(f"Pushing {dataset_name} to hub")
        dataset.push_to_hub(dataset_name)


if __name__ == "__main__":
    create_sft_dataset_with_n_outputs([1, 4, 16, 32])
