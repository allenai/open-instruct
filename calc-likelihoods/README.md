# Response Log Likelihood Calculator

This toolkit calculates the average log likelihood of responses given prompts across multiple language models, useful for comparing model confidence and quality over time.

## Overview

The main functionality:
1. **Applies chat templates** to format prompts and responses appropriately for each model
2. **Calculates log likelihoods** for response tokens only (excluding prompt tokens)
3. **Compares multiple models** to see how response likelihood changes
4. **Generates visualizations** and reports for analysis

## Key Features

- **Proper token separation**: Only calculates likelihood for response tokens, not prompt tokens
- **Chat template support**: Automatically applies model-specific chat templates
- **Multiple model comparison**: Iterate over any list of HuggingFace models
- **Comprehensive metrics**: Includes average log likelihood, perplexity, and token counts
- **Visualization**: Generates plots and comparison reports

## Installation

```bash
pip install torch transformers datasets pandas matplotlib seaborn tqdm --break-system-packages
```

## Quick Start

### Basic Usage

```python
from datasets import Dataset
from calculate_response_loglikelihoods import evaluate_models_on_dataset

# Prepare your data
data = [
    {
        "prompt": "What is the capital of France?",
        "response": "The capital of France is Paris."
    },
    {
        "prompt": "Explain photosynthesis briefly.",
        "response": "Photosynthesis converts sunlight into energy in plants."
    }
]

dataset = Dataset.from_list(data)

# Define models to compare
model_names = [
    "gpt2",
    "gpt2-medium",
    "gpt2-large"
]

# Run evaluation
results = evaluate_models_on_dataset(
    dataset=dataset,
    model_names=model_names,
    prompt_column="prompt",
    response_column="response"
)

# Save results
results.to_csv("results.csv", index=False)
```

### Loading Data from Different Sources

#### From CSV
```python
import pandas as pd
from datasets import Dataset

df = pd.read_csv("your_data.csv")
dataset = Dataset.from_pandas(df)
```

#### From JSONL
```python
import json
from datasets import Dataset

data = []
with open("your_data.jsonl", 'r') as f:
    for line in f:
        data.append(json.loads(line))
dataset = Dataset.from_list(data)
```

#### From HuggingFace Hub
```python
from datasets import load_dataset

dataset = load_dataset("your_dataset_name", split="test")
```

## Understanding the Metrics

### Average Log Likelihood
- **Higher is better** (closer to 0, less negative)
- Represents how "expected" the response was according to the model
- Formula: `sum(log_prob(token_i)) / num_response_tokens`

### Perplexity
- **Lower is better**
- Measures how "surprised" the model is by the response
- Formula: `exp(-average_log_likelihood)`
- More interpretable than raw log likelihood

### Number of Response Tokens
- Count of tokens in the response only (prompt excluded)
- Useful for normalizing comparisons

## How It Works

### 1. Token Separation
The script carefully separates prompt tokens from response tokens:

```python
# Full conversation
messages = [
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI stands for Artificial Intelligence."}
]

# Tokenize full conversation
full_tokens = tokenizer.apply_chat_template(messages, ...)

# Tokenize just the prompt to find where response starts
prompt_only = tokenizer.apply_chat_template([messages[0]], ...)
prompt_length = len(prompt_only)

# Calculate log likelihood only for tokens after prompt_length
```

### 2. Log Likelihood Calculation
For each token in the response:
1. Get model's predicted probability distribution
2. Extract the log probability of the actual token
3. Average across all response tokens

```python
# For each response token
log_probs = log_softmax(model_logits, dim=-1)
token_log_prob = log_probs[position, actual_token_id]

# Average
avg_log_likelihood = mean(all_token_log_probs)
```

## Comparing Models Over Time

This is particularly useful for:
- **Analyzing model iterations** (e.g., GPT-2 → GPT-2 Medium → GPT-2 Large)
- **Comparing different model families** (e.g., Llama vs Mistral vs GPT)
- **Tracking training progress** (checkpoints at different steps)

Example model lists:

```python
# Comparing model sizes
models = [
    "gpt2",           # 124M params
    "gpt2-medium",    # 355M params
    "gpt2-large",     # 774M params
    "gpt2-xl"         # 1.5B params
]

# Comparing training checkpoints
models = [
    "your-model-checkpoint-1000",
    "your-model-checkpoint-2000",
    "your-model-checkpoint-3000",
    "your-model-final"
]

# Comparing model families
models = [
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "google/gemma-7b-it"
]
```

## Output Files

### results.csv
Contains detailed results for each sample and model:
- `model`: Model name
- `sample_id`: Sample index
- `prompt`: The input prompt
- `response`: The response being evaluated
- `avg_log_likelihood`: Average log likelihood (higher/less negative is better)
- `num_response_tokens`: Number of tokens in response
- `perplexity`: Perplexity score (lower is better)

### Visualizations
- **log_likelihood_comparison.png**: Box plots and bar charts
- **perplexity_comparison.png**: Perplexity distribution
- **log_likelihood_heatmap.png**: Sample-wise comparison
- **log_likelihood_trends.png**: Trends across samples

### comparison_report.txt
Text summary with:
- Overall statistics per model
- Model ranking
- Best model per sample

## Advanced Usage

### Custom Column Names
```python
results = evaluate_models_on_dataset(
    dataset=dataset,
    model_names=models,
    prompt_column="question",  # Your prompt column name
    response_column="answer"   # Your response column name
)
```

### GPU/CPU Selection
```python
# Force CPU
results = evaluate_models_on_dataset(
    dataset=dataset,
    model_names=models,
    device="cpu"
)

# Use GPU (default if available)
results = evaluate_models_on_dataset(
    dataset=dataset,
    model_names=models,
    device="cuda"
)
```

### Processing Large Datasets
For large datasets, consider:

```python
# Process in batches
from datasets import Dataset

full_dataset = Dataset.from_pandas(large_df)

# Process 100 samples at a time
batch_size = 100
for i in range(0, len(full_dataset), batch_size):
    batch = full_dataset.select(range(i, min(i + batch_size, len(full_dataset))))
    results = evaluate_models_on_dataset(batch, models)
    results.to_csv(f"results_batch_{i}.csv", index=False)
```

## Interpretation Guide

### What makes a "good" log likelihood?
- **Context matters**: Different tasks have different expected ranges
- **Relative comparison**: Focus on comparing models rather than absolute values
- **Higher is better**: -2.5 is better than -5.0

### Typical ranges:
- **Very confident**: -0.5 to -1.5
- **Confident**: -1.5 to -3.0
- **Uncertain**: -3.0 to -5.0
- **Very uncertain**: < -5.0

### Red flags:
- **Extreme values**: < -10.0 might indicate issues
- **High variance**: Inconsistent model behavior
- **Unexpected ranking**: Smaller model outperforming larger one might indicate data issues

## Troubleshooting

### Out of Memory
```python
# Use smaller models
# Process fewer samples at once
# Use CPU instead of GPU
# Use float16 precision (already default on GPU)
```

### Model Not Found
```python
# Check model name spelling
# Ensure you have access (some models require authentication)
# Try downloading the model first:
from transformers import AutoModel
AutoModel.from_pretrained("model-name", use_auth_token=True)
```

### Chat Template Not Available
The script automatically falls back to simple concatenation if chat templates aren't available.

## Performance Tips

1. **Use GPU**: Much faster for large models
2. **Batch processing**: Process large datasets in chunks
3. **Model caching**: Models are cached after first download
4. **Limit samples**: Start with a small subset to test

## Example Workflow

```bash
# 1. Prepare your data
# data.csv with columns: prompt, response

# 2. Run evaluation
python example_usage.py

# 3. Check outputs
ls /mnt/user-data/outputs/
# - log_likelihood_results.csv
# - log_likelihood_comparison.png
# - perplexity_comparison.png
# - log_likelihood_heatmap.png
# - log_likelihood_trends.png
# - comparison_report.txt

# 4. Analyze results
cat /mnt/user-data/outputs/comparison_report.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{response_log_likelihood_calculator,
  title = {Response Log Likelihood Calculator},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/repo}
}
```

## License

MIT License - feel free to use and modify!
