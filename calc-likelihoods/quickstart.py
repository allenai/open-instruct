#!/usr/bin/env python3
"""
QUICKSTART SCRIPT
Run this to test the log likelihood calculator with minimal setup.
"""

from datasets import Dataset
from calculate_response_loglikelihoods import evaluate_models_on_dataset
import pandas as pd

def main():
    print("\n" + "="*80)
    print("LOG LIKELIHOOD CALCULATOR - QUICKSTART")
    print("="*80 + "\n")
    
    # Simple test data
    print("Creating test dataset...")
    test_data = [
        {
            "prompt": "What is 2+2?",
            "response": "The answer is 4."
        },
        {
            "prompt": "What color is the sky?",
            "response": "The sky is blue."
        },
        {
            "prompt": "What is the capital of France?",
            "response": "Paris is the capital of France."
        }
    ]
    
    dataset = Dataset.from_list(test_data)
    print(f"✓ Created dataset with {len(test_data)} samples\n")
    
    # Use smallest available model for quick testing
    print("Testing with GPT-2 (smallest model for speed)...")
    model_names = ["gpt2"]
    
    print("\nRunning evaluation...")
    print("Note: First run will download the model (~500MB)\n")
    
    # Run evaluation
    results_df = evaluate_models_on_dataset(
        dataset=dataset,
        model_names=model_names,
        prompt_column="prompt",
        response_column="response"
    )
    
    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80 + "\n")
    
    print(results_df[['sample_id', 'model', 'avg_log_likelihood', 'perplexity', 'num_response_tokens']])
    
    # Save results
    output_path = "/mnt/user-data/outputs/quickstart_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to: {output_path}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    mean_ll = results_df['avg_log_likelihood'].mean()
    mean_perplexity = results_df['perplexity'].mean()
    print(f"Mean Log Likelihood: {mean_ll:.4f}")
    print(f"Mean Perplexity: {mean_perplexity:.4f}")
    
    print("\n" + "="*80)
    print("SUCCESS! 🎉")
    print("="*80)
    print("\nNext steps:")
    print("1. Modify this script with your own data")
    print("2. Add more models to compare: ['gpt2', 'gpt2-medium', 'gpt2-large']")
    print("3. Use example_usage.py for visualizations")
    print("4. Check README.md for detailed documentation")
    print("\n")


if __name__ == "__main__":
    main()
