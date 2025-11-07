import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_response_log_likelihood(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[float, int]:
    """
    Calculate the average log likelihood of a response given a prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        response: The response to evaluate
        device: Device to run on
        
    Returns:
        Tuple of (average_log_likelihood, num_response_tokens)
    """
    # Format as a conversation
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    full_text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Tokenize the full text
    full_tokens = tokenizer(full_text, return_tensors="pt").input_ids.to(device)
    
    # Tokenize just the prompt to find where response starts
    messages_prompt = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(
        messages_prompt, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    prompt_tokens = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
    prompt_length = prompt_tokens.shape[1]
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(full_tokens, labels=full_tokens)
        logits = outputs.logits
    
    # Calculate log probabilities
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Get the log probabilities of the actual tokens
    # Note: logits[0, i] predicts token[i+1]
    token_log_probs = []
    for i in range(prompt_length - 1, full_tokens.shape[1] - 1):
        token_id = full_tokens[0, i + 1].item()
        token_log_prob = log_probs[0, i, token_id].item()
        token_log_probs.append(token_log_prob)
    
    # Calculate average
    if len(token_log_probs) > 0:
        avg_log_likelihood = np.mean(token_log_probs)
        num_response_tokens = len(token_log_probs)
    else:
        avg_log_likelihood = 0.0
        num_response_tokens = 0
    
    return avg_log_likelihood, num_response_tokens


def evaluate_models_on_dataset(
    dataset: Dataset,
    model_names: List[str],
    prompt_column: str = "prompt",
    response_column: str = "response",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1  # Processing one at a time for simplicity
) -> pd.DataFrame:
    """
    Evaluate multiple models on a dataset and calculate response log likelihoods.
    
    Args:
        dataset: HuggingFace dataset or list of dicts with prompt/response pairs
        model_names: List of model identifiers
        prompt_column: Name of the prompt column
        response_column: Name of the response column
        device: Device to run on
        batch_size: Batch size (currently only supports 1)
        
    Returns:
        DataFrame with results for each model and each sample
    """
    results = []
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_1000/")


    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Loading model: {model_name}")
        print(f"{'='*80}")
        
        try:            
            # Set padding token if not set
            # if tokenizer.pad_token is None:
                # tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                model = model.to(device)
            
            model.eval()
            
            # Process each sample
            for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {model_name}")):
                messages = sample["messages"]
                prompt = messages[0]["content"]
                response = messages[0]["content"]
                
                try:
                    avg_log_likelihood, num_tokens = calculate_response_log_likelihood(
                        model, tokenizer, prompt, response, device
                    )
                    
                    results.append({
                        "model": model_name,
                        "sample_id": idx,
                        "prompt": prompt,
                        "response": response,
                        "avg_log_likelihood": avg_log_likelihood,
                        "num_response_tokens": num_tokens,
                        "perplexity": np.exp(-avg_log_likelihood) if avg_log_likelihood != 0 else float('inf')
                    })
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    results.append({
                        "model": model_name,
                        "sample_id": idx,
                        "prompt": prompt,
                        "response": response,
                        "avg_log_likelihood": None,
                        "num_response_tokens": 0,
                        "perplexity": None
                    })
            
            # Clean up
            del model
            # del tokenizer
            torch.cuda.empty_cache() if device == "cuda" else None
            
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            continue
    
    return pd.DataFrame(results)

def load_data_from_huggingface(dataset_name: str, split: str = "test", subset: str = None) -> Dataset:
    """Load data from HuggingFace datasets hub"""
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    return dataset

def main():
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
    
    # Run evaluation
    results_df = evaluate_models_on_dataset(
        dataset=dataset,
        model_names=model_names,
        prompt_column="prompt",
        response_column="response"
    )
    
    # Save results
    results_df.to_csv("/mnt/user-data/outputs/log_likelihood_results.csv", index=False)
    print("\n" + "="*80)
    print("Results saved to: /mnt/user-data/outputs/log_likelihood_results.csv")
    print("="*80)
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL")
    print("="*80)
    summary = results_df.groupby('model').agg({
        'avg_log_likelihood': ['mean', 'std', 'min', 'max'],
        'perplexity': ['mean', 'std', 'min', 'max'],
        'num_response_tokens': 'mean'
    }).round(4)
    print(summary)
    
    # Display individual results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    for model in model_names:
        print(f"\nModel: {model}")
        model_results = results_df[results_df['model'] == model]
        print(model_results[['sample_id', 'avg_log_likelihood', 'perplexity', 'num_response_tokens']])


if __name__ == "__main__":
    main()
