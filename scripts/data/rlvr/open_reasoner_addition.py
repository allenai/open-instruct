import json
import argparse
import random
from pathlib import Path
import datasets
from huggingface_hub import HfApi
import gmpy2
from tqdm import tqdm

random.seed(0)


import gmpy2
from tqdm import tqdm
import itertools


def generate_random_examples(num_digit, sample_size):
    """Generate unique random numbers with the specified digit length."""
    # Calculate start and end for the given number of digits
    start = 10 ** (num_digit - 1)  # Smallest number with num_digit digits
    end = 10 ** num_digit - 1      # Largest number with num_digit digits

    # Ensure sufficient randomness and uniqueness
    unique_numbers = set()
    attempts = 0
    max_attempts = sample_size * 10  # Prevent infinite loop

    while len(unique_numbers) < sample_size and attempts < max_attempts:
        # Use getrandbits to generate large numbers
        # We'll use the number's bit length to ensure it fits the digit requirement
        num = random.getrandbits(num_digit * 4)  # Multiply by 4 to ensure enough bits
        
        # Constrain the number to the correct digit range
        num = (num % (end - start + 1)) + start
        
        unique_numbers.add(num)
        attempts += 1

    # If not enough unique numbers, print a warning
    if len(unique_numbers) < sample_size:
        print(f"⚠️ Could only generate {len(unique_numbers)} unique numbers for {num_digit} digits")

    return list(unique_numbers) 



def generate_large_number_addition_data(a_max, b_max, n_train, n_test):
    """Generate train and test datasets for addition of large-digit numbers."""
    datasets_dict = {}

    for a_digits in range(1, a_max + 1):
        for b_digits in range(1, b_max + 1):
            dataset_name = f"{a_digits}x{b_digits}"
            
            print(f"\n📌 Generating dataset: {dataset_name}")

            # Generate `a_digits` and `b_digits` length numbers
            a_numbers = generate_random_examples(a_digits, n_train + n_test)
            b_numbers = generate_random_examples(b_digits, n_train + n_test)

            # Create all possible pairs
            all_pairs = list(itertools.product(a_numbers, b_numbers))
            random.shuffle(all_pairs)

            # Split into train and test
            train_data = all_pairs[:n_train]
            test_data = all_pairs[n_train: n_train+n_test]

            datasets_dict[dataset_name] = {"train": train_data, "test": test_data}

    return datasets_dict

def save_to_jsonl(data, file_path, split_name):
    """Save dataset to JSONL format with an explicit split."""
    with open(file_path, "w") as f, tqdm(total=len(data), desc=f"Saving {split_name} data") as pbar:
        for a, b in data:
            json.dump({"problem": f"What is {a} plus {b}?", "answer": str(a + b), "split": split_name}, f)
            f.write("\n")
            pbar.update(1)

def prepare_datasets(output_dir):
    """Prepare train and test datasets ensuring each addition group is saved separately."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_datasets = generate_large_number_addition_data(a_max=15, b_max=15, n_train=1000, n_test=100)

    saved_files = {}
    for name, data in tqdm(all_datasets.items(), desc="Saving all datasets"):
        len_train = len(data["train"])
        len_test = len(data["test"])
        train_file = output_dir / f"addition_{len_train}_train_{name}.jsonl"
        test_file = output_dir / f"addition_{len_test}_test_{name}.jsonl"

        save_to_jsonl(data["train"], train_file, "train")
        save_to_jsonl(data["test"], test_file, "test")

        saved_files[name] = {"train": train_file, "test": test_file}

    print(f"\n✅ Datasets saved locally under {output_dir}")
    return saved_files  # Dictionary with all saved file paths

def process_file(file_path, split_name):
    """Convert JSONL data into Hugging Face dataset format with explicit splits."""
    with open(file_path, "r") as f:
        data = [json.loads(line.strip()) for line in f if line.strip()]

    dataset = {
        "messages": [[
            {"role": "user", "content": item["problem"]},
            {"role": "assistant", "content": item["answer"]},
        ] for item in data],
        "ground_truth": [item["answer"] for item in data],
        "dataset": ["addition"] * len(data),
        "split": [split_name] * len(data),
    }
    return datasets.Dataset.from_dict(dataset)

def push_to_huggingface(saved_files, hf_entity):
    """Push each addition dataset (e.g., `9x5`, `3x4`) separately to Hugging Face."""
    api = HfApi()
    hf_entity = hf_entity or api.whoami()["name"]

    print("\n📤 Uploading each addition dataset separately to Hugging Face...\n")

    for name, files in tqdm(saved_files.items(), desc="Uploading datasets"):
        train_data = process_file(files["train"], "train")
        test_data = process_file(files["test"], "test")

        dataset_dict = datasets.DatasetDict({
            "train": train_data,
            "test": test_data,
        })

        repo_id = f"{hf_entity}/addition_{name}"
        dataset_dict.push_to_hub(repo_id)

        print(f"✅ Dataset `{name}` uploaded successfully to {repo_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/weka/oe-adapt-default/nouhad/data/math_data/addition", help="Output directory")
    parser.add_argument("--push_to_hub", action="store_true", help="Upload to Hugging Face")
    parser.add_argument("--hf_entity", type=str, default=None, help="Hugging Face entity")
    args = parser.parse_args()

    saved_files = prepare_datasets(args.output_dir)

    if args.push_to_hub:
        push_to_huggingface(saved_files, args.hf_entity)

if __name__ == "__main__":
    main()
