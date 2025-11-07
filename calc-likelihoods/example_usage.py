"""
Example script showing how to use the log likelihood calculator
with custom data sources and create visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset, load_dataset
from calculate_response_loglikelihoods import evaluate_models_on_dataset


def load_data_from_csv(filepath: str) -> Dataset:
    """Load data from CSV file"""
    df = pd.read_csv(filepath)
    return Dataset.from_pandas(df)


def load_data_from_jsonl(filepath: str) -> Dataset:
    """Load data from JSONL file"""
    import json
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return Dataset.from_list(data)


def load_data_from_huggingface(dataset_name: str, split: str = "test", subset: str = None) -> Dataset:
    """Load data from HuggingFace datasets hub"""
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset


def visualize_results(results_df: pd.DataFrame, output_dir: str = "/mnt/user-data/outputs"):
    """Create visualizations of the results"""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # 1. Average log likelihood by model
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Box plot of log likelihoods
    ax1 = axes[0]
    results_df.boxplot(column='avg_log_likelihood', by='model', ax=ax1)
    ax1.set_title('Distribution of Average Log Likelihoods by Model')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Average Log Likelihood')
    plt.sca(ax1)
    plt.xticks(rotation=45, ha='right')
    
    # Bar plot of mean log likelihoods
    ax2 = axes[1]
    mean_ll = results_df.groupby('model')['avg_log_likelihood'].mean().sort_values()
    mean_ll.plot(kind='barh', ax=ax2)
    ax2.set_title('Mean Average Log Likelihood by Model')
    ax2.set_xlabel('Average Log Likelihood')
    ax2.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/log_likelihood_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/log_likelihood_comparison.png")
    plt.close()
    
    # 2. Perplexity comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    results_df.boxplot(column='perplexity', by='model', ax=ax)
    ax.set_title('Distribution of Perplexity by Model')
    ax.set_xlabel('Model')
    ax.set_ylabel('Perplexity')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/perplexity_comparison.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/perplexity_comparison.png")
    plt.close()
    
    # 3. Heatmap of log likelihoods per sample
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_table = results_df.pivot(index='sample_id', columns='model', values='avg_log_likelihood')
    sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax)
    ax.set_title('Average Log Likelihood Heatmap (per sample and model)')
    ax.set_xlabel('Model')
    ax.set_ylabel('Sample ID')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/log_likelihood_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/log_likelihood_heatmap.png")
    plt.close()
    
    # 4. Line plot showing trends across samples
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model].sort_values('sample_id')
        ax.plot(model_data['sample_id'], model_data['avg_log_likelihood'], marker='o', label=model)
    
    ax.set_title('Average Log Likelihood Across Samples')
    ax.set_xlabel('Sample ID')
    ax.set_ylabel('Average Log Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/log_likelihood_trends.png", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir}/log_likelihood_trends.png")
    plt.close()


def create_comparison_report(results_df: pd.DataFrame, output_path: str = "/mnt/user-data/outputs/comparison_report.txt"):
    """Create a text report comparing models"""
    
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-"*80 + "\n")
        for model in results_df['model'].unique():
            model_data = results_df[results_df['model'] == model]
            f.write(f"\nModel: {model}\n")
            f.write(f"  Mean Log Likelihood: {model_data['avg_log_likelihood'].mean():.4f}\n")
            f.write(f"  Std Log Likelihood:  {model_data['avg_log_likelihood'].std():.4f}\n")
            f.write(f"  Mean Perplexity:     {model_data['perplexity'].mean():.4f}\n")
            f.write(f"  Min Log Likelihood:  {model_data['avg_log_likelihood'].min():.4f}\n")
            f.write(f"  Max Log Likelihood:  {model_data['avg_log_likelihood'].max():.4f}\n")
        
        # Ranking
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL RANKING (by mean log likelihood)\n")
        f.write("-"*80 + "\n")
        ranking = results_df.groupby('model')['avg_log_likelihood'].mean().sort_values(ascending=False)
        for rank, (model, score) in enumerate(ranking.items(), 1):
            f.write(f"{rank}. {model}: {score:.4f}\n")
        
        # Per-sample winner
        f.write("\n" + "="*80 + "\n")
        f.write("BEST MODEL PER SAMPLE\n")
        f.write("-"*80 + "\n")
        for sample_id in results_df['sample_id'].unique():
            sample_data = results_df[results_df['sample_id'] == sample_id]
            best_model = sample_data.loc[sample_data['avg_log_likelihood'].idxmax(), 'model']
            best_score = sample_data['avg_log_likelihood'].max()
            f.write(f"Sample {sample_id}: {best_model} ({best_score:.4f})\n")
    
    print(f"\nReport saved to: {output_path}")


# Example usage configurations
def example_1_simple_list():
    """Example: Using a simple list of prompts and responses"""
    data = [
        {"prompt": "What is AI?", "response": "AI stands for Artificial Intelligence."},
        {"prompt": "What is ML?", "response": "ML stands for Machine Learning."},
    ]
    
    dataset = Dataset.from_list(data)
    models = ["gpt2"]  # Add your models
    
    results = evaluate_models_on_dataset(dataset, models)
    return results


def example_2_from_csv():
    """Example: Loading data from CSV file"""
    # Your CSV should have 'prompt' and 'response' columns
    # dataset = load_data_from_csv("/path/to/your/data.csv")
    # models = ["model1", "model2", "model3"]
    # results = evaluate_models_on_dataset(dataset, models)
    # return results
    pass


def example_3_from_huggingface():
    """Example: Loading from HuggingFace datasets"""
    # Example with a small subset for testing
    # dataset = load_data_from_huggingface("squad", split="validation[:10]")
    # You may need to preprocess to create 'prompt' and 'response' columns
    # results = evaluate_models_on_dataset(dataset, models, prompt_column="question", response_column="answers")
    # return results
    pass


def main():
    """Main execution"""
    
    # Choose your data source
    print("Loading data...")
    
    # # Option 1: Simple example
    # sample_data = [
    #     {"prompt": "What is Python?", "response": "Python is a programming language."},
    #     {"prompt": "What is JavaScript?", "response": "JavaScript is a programming language for web development."},
    #     {"prompt": "What is SQL?", "response": "SQL is a language for managing databases."},
    # ]
    # dataset = Dataset.from_list(sample_data)

    dataset = load_data_from_huggingface("jacobmorrison/social-rl-eval-dataset-100", split="train")
    
    # Option 2: Load from CSV (uncomment and modify)
    # dataset = load_data_from_csv("/mnt/user-data/uploads/your_data.csv")
    
    # Option 3: Load from JSONL (uncomment and modify)
    # dataset = load_data_from_jsonl("/mnt/user-data/uploads/your_data.jsonl")
    
    # Define your models
    # Replace with your actual model names
    model_names = [
        "allenai/Olmo-3-1025-7B",
        # Add your models here
    ]
    
    print(f"Evaluating {len(model_names)} models on {len(dataset)} samples...")
    
    # Run evaluation
    results_df = evaluate_models_on_dataset(
        dataset=dataset,
        model_names=model_names,
        prompt_column="prompt",
        response_column="response"
    )
    
    # Save results
    results_df.to_csv("/mnt/user-data/outputs/log_likelihood_results.csv", index=False)
    print("\nResults saved!")
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_results(results_df)
    
    # Create comparison report
    print("\nCreating comparison report...")
    create_comparison_report(results_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print("  - log_likelihood_results.csv")
    print("  - log_likelihood_comparison.png")
    print("  - perplexity_comparison.png")
    print("  - log_likelihood_heatmap.png")
    print("  - log_likelihood_trends.png")
    print("  - comparison_report.txt")


if __name__ == "__main__":
    main()
