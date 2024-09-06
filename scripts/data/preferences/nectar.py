from datasets import load_dataset, Dataset
import random
import argparse

def load_nectar_dataset(subset=None, deduplication=False):
    # Load the dataset
    dataset = load_dataset("berkeley-nest/Nectar", split="train")
    print(f"Original dataset size: {len(dataset)}")
    
    # Filter the dataset based on the subset
    filtered_dataset = dataset.filter(lambda example: example['source'] == [subset])
    print(f"Filtered dataset size: {len(filtered_dataset)}")
    
    binarized_data = []
    
    for example in filtered_dataset:
        prompt = example['prompt']
        answers = example['answers']
        
        # Sort answers by rank (lower rank is better)
        sorted_answers = sorted(answers, key=lambda x: x['rank'])
        
        if len(sorted_answers) >= 2:
            # Randomly select the chosen answer from the top 4 (or all if less than 4)
            top_n = min(4, len(sorted_answers))
            chosen = random.choice(sorted_answers[:top_n])
            
            # Select the rejected answer from the remaining answers
            remaining_answers = [a for a in sorted_answers if a != chosen]
            rejected = random.choice(remaining_answers)
            
            binarized_example = {
                "prompt": prompt,
                "chosen": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": chosen['answer']}
                ],
                "rejected": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected['answer']}
                ],
                "subset": example['source'],
            }
            
            binarized_data.append(binarized_example)
    
    print(f"Number of binarized examples: {len(binarized_data)}")
    
    # TODO: Implement deduplication logic here when needed
    if deduplication:
        pass  # Placeholder for future implementation
    
    return binarized_data

def main():
    parser = argparse.ArgumentParser(description="Load and process the Nectar dataset")
    parser.add_argument("--dataset_name", type=str, default="nectar_binarized", help="Name for the processed dataset")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the dataset to Hugging Face Hub")
    parser.add_argument("--hf_entity", type=str, default=None, help="Hugging Face username or organization")
    parser.add_argument("--deduplication", action="store_true", help="Apply deduplication to the dataset")
    parser.add_argument("--subset", type=str, default="lmsys-chat-1m", help="Subset of the dataset to use")
    
    args = parser.parse_args()

    # Load and process the dataset
    binarized_data = load_nectar_dataset(subset=args.subset, deduplication=args.deduplication)
    
    # Convert to Hugging Face Dataset
    binarized_ds = Dataset.from_list(binarized_data)

    # Dataset naming logic
    dataset_name = args.dataset_name
    if args.deduplication:
        dataset_name += "-dedup-ultrafeedback"
    else:
        dataset_name += f"-{args.subset}"

    # Saving logic
    if args.push_to_hub:
        if args.hf_entity:
            full_name = f"{args.hf_entity}/{dataset_name}"
        else:
            full_name = dataset_name

        print(f"Pushing dataset to Hub: {full_name}")
        binarized_ds.push_to_hub(full_name, private=True)
    else:
        print(f"Dataset {dataset_name} created locally (not pushed to Hub)")
        # Optionally, you can save the dataset locally here
        # binarized_ds.save_to_disk(dataset_name)

    print("First example:")
    print(binarized_data[0])

if __name__ == "__main__":
    main()