from datasets import load_dataset
import pandas as pd

def find_prompt_overlap():
    # Load both datasets
    print("Loading datasets...")
    qwq_dataset = load_dataset("allenai/preference-qwq-judge", split="train")
    gemma_dataset = load_dataset("allenai/preference-gemma3-judge", split="train")
    
    # Convert to pandas for faster filtering and set operations
    print("Converting to pandas and filtering...")
    qwq_df = qwq_dataset.to_pandas()
    gemma_df = gemma_dataset.to_pandas()
    
    # Filter for valid rows and get unique prompt_ids
    qwq_valid_prompts = set(qwq_df[qwq_df['is_valid_row'] == True]['prompt_id'].unique())
    gemma_valid_prompts = set(gemma_df[gemma_df['is_valid_row'] == True]['prompt_id'].unique())
    
    # Find overlap
    overlap = qwq_valid_prompts.intersection(gemma_valid_prompts)
    
    # Print results
    print(f"\nResults:")
    print(f"QWQ valid prompt_ids: {len(qwq_valid_prompts)}")
    print(f"Gemma valid prompt_ids: {len(gemma_valid_prompts)}")
    print(f"Overlapping prompt_ids: {len(overlap)}")
    print(f"Overlap percentage: {len(overlap) / min(len(qwq_valid_prompts), len(gemma_valid_prompts)) * 100:.2f}%")
    
    return overlap

if __name__ == "__main__":
    overlapping_prompts = find_prompt_overlap()
    print(f"\nFirst 10 overlapping prompt_ids: {list(overlapping_prompts)[:10]}")