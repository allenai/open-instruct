#!/usr/bin/env python3
"""
CONFIGURATION TEMPLATE
Copy this file and modify it for your specific use case.
"""

from datasets import Dataset, load_dataset
from calculate_response_loglikelihoods import evaluate_models_on_dataset
import pandas as pd

# ============================================================================
# CONFIGURATION - MODIFY THESE SETTINGS
# ============================================================================

# 1. DATA SOURCE
# Choose ONE of the following methods to load your data:

# Option A: Load from a list
USE_LIST = False
DATA_LIST = [
    {"prompt": "Your prompt here", "response": "Your response here"},
    {"prompt": "Another prompt", "response": "Another response"},
]

# Option B: Load from CSV file
USE_CSV = False
CSV_PATH = "/mnt/user-data/uploads/your_data.csv"
PROMPT_COLUMN = "prompt"  # Name of your prompt column
RESPONSE_COLUMN = "response"  # Name of your response column

# Option C: Load from JSONL file
USE_JSONL = False
JSONL_PATH = "/mnt/user-data/uploads/your_data.jsonl"

# Option D: Load from HuggingFace dataset
USE_HUGGINGFACE = True  # DEFAULT
HF_DATASET_NAME = "squad"  # Example: "squad", "glue", etc.
HF_SPLIT = "validation[:10]"  # Example: "train", "test", "validation[:100]"
HF_SUBSET = None  # Some datasets need a subset, e.g., "sst2" for "glue"

# 2. MODELS TO COMPARE
# List the models you want to evaluate
# Examples:
# - Size progression: ["gpt2", "gpt2-medium", "gpt2-large"]
# - Different families: ["gpt2", "EleutherAI/pythia-160m", "facebook/opt-125m"]
# - Your checkpoints: ["your-model-step-1000", "your-model-step-2000"]

MODELS = [
    "gpt2",  # Start with one model for testing
    # "gpt2-medium",
    # "gpt2-large",
    # Add your models here
]

# 3. PROCESSING OPTIONS
DEVICE = "cuda"  # Use "cuda" for GPU or "cpu" for CPU
OUTPUT_DIR = "/mnt/user-data/outputs"

# 4. VISUALIZATION OPTIONS
CREATE_VISUALIZATIONS = True
CREATE_REPORT = True

# ============================================================================
# MAIN EXECUTION - NO NEED TO MODIFY BELOW THIS LINE
# ============================================================================

def load_data():
    """Load data based on configuration"""
    if USE_LIST:
        print("Loading data from list...")
        return Dataset.from_list(DATA_LIST), PROMPT_COLUMN, RESPONSE_COLUMN
    
    elif USE_CSV:
        print(f"Loading data from CSV: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        return Dataset.from_pandas(df), PROMPT_COLUMN, RESPONSE_COLUMN
    
    elif USE_JSONL:
        print(f"Loading data from JSONL: {JSONL_PATH}")
        import json
        data = []
        with open(JSONL_PATH, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return Dataset.from_list(data), PROMPT_COLUMN, RESPONSE_COLUMN
    
    elif USE_HUGGINGFACE:
        print(f"Loading data from HuggingFace: {HF_DATASET_NAME}")
        if HF_SUBSET:
            dataset = load_dataset(HF_DATASET_NAME, HF_SUBSET, split=HF_SPLIT)
        else:
            dataset = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
        
        # For demo purposes with SQuAD, we'll create prompt/response from question/context
        # MODIFY THIS SECTION based on your dataset structure
        print("Note: You may need to preprocess the dataset to create prompt/response columns")
        return dataset, "question", "context"  # Adjust based on your dataset
    
    else:
        raise ValueError("No data source selected! Set one of USE_LIST, USE_CSV, USE_JSONL, or USE_HUGGINGFACE to True")


def main():
    print("\n" + "="*80)
    print("LOG LIKELIHOOD EVALUATION")
    print("="*80 + "\n")
    
    # Load data
    dataset, prompt_col, response_col = load_data()
    print(f"✓ Loaded {len(dataset)} samples\n")
    
    # Display sample
    print("Sample data:")
    print(f"  Prompt column: {prompt_col}")
    print(f"  Response column: {response_col}")
    if len(dataset) > 0:
        print(f"  Example: {dataset[0]}\n")
    
    # Run evaluation
    print(f"Evaluating {len(MODELS)} model(s)...")
    print(f"Models: {', '.join(MODELS)}\n")
    
    results_df = evaluate_models_on_dataset(
        dataset=dataset,
        model_names=MODELS,
        prompt_column=prompt_col,
        response_column=response_col,
        device=DEVICE
    )
    
    # Save results
    results_path = f"{OUTPUT_DIR}/log_likelihood_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Create visualizations
    if CREATE_VISUALIZATIONS and len(MODELS) > 1:
        print("\nCreating visualizations...")
        from example_usage import visualize_results
        visualize_results(results_df, OUTPUT_DIR)
    
    # Create report
    if CREATE_REPORT:
        print("\nCreating comparison report...")
        from example_usage import create_comparison_report
        create_comparison_report(results_df, f"{OUTPUT_DIR}/comparison_report.txt")
    
    # Display summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    summary = results_df.groupby('model').agg({
        'avg_log_likelihood': ['mean', 'std'],
        'perplexity': ['mean', 'std'],
        'num_response_tokens': 'mean'
    }).round(4)
    print(summary)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE! 🎉")
    print("="*80)
    print(f"\nCheck {OUTPUT_DIR}/ for all output files.\n")


if __name__ == "__main__":
    main()
