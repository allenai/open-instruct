import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_response_entropy(
    model,
    tokenizer,
    prompt: str,
    response: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[float, float, int]:
    """
    Calculate the average entropy of the model's output distribution over response tokens.
    
    Entropy measures the model's uncertainty at each token position.
    Higher entropy = more uncertainty, lower entropy = more confident predictions.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        response: The response to evaluate
        device: Device to run on
        
    Returns:
        Tuple of (average_entropy, total_entropy, num_response_tokens)
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
        outputs = model(full_tokens)
        logits = outputs.logits
    
    # Calculate probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    
    # Calculate entropy at each position: H = -sum(p * log(p))
    # Note: logits[0, i] predicts token[i+1], so we look at positions predicting response tokens
    token_entropies = []
    for i in range(prompt_length - 1, full_tokens.shape[1] - 1):
        # Get the probability distribution at this position
        p = probs[0, i]  # Shape: [vocab_size]
        log_p = log_probs[0, i]  # Shape: [vocab_size]
        
        # Entropy: H = -sum(p * log(p))
        # Using natural log (base e), multiply by log2(e) to convert to bits if needed
        entropy = -torch.sum(p * log_p).item()
        token_entropies.append(entropy)
    
    # Calculate statistics
    if len(token_entropies) > 0:
        avg_entropy = np.mean(token_entropies)
        total_entropy = np.sum(token_entropies)
        num_response_tokens = len(token_entropies)
    else:
        avg_entropy = 0.0
        total_entropy = 0.0
        num_response_tokens = 0
    
    return avg_entropy, total_entropy, num_response_tokens


def evaluate_models_on_dataset(
    dataset: Dataset,
    model_names: List[str],
    prompt_column: str = "prompt",
    response_column: str = "response",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1  # Processing one at a time for simplicity
) -> pd.DataFrame:
    """
    Evaluate multiple models on a dataset and calculate response entropy.
    
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
        if "Olmo" in model_name:
            save_name = "step_0"
        else:
            save_name = model_name.replace("/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/", "").strip()
        print(f"\n{'='*80}")
        print(f"Loading model: {model_name}")
        print(f"Save name: {save_name}")
        print(f"{'='*80}")
        
        try:            
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
                response = messages[1]["content"]  # Fixed: should be index 1 for assistant response
                
                try:
                    avg_entropy, total_entropy, num_tokens = calculate_response_entropy(
                        model, tokenizer, prompt, response, device
                    )
                    
                    results.append({
                        "model": save_name,
                        "sample_id": idx,
                        "prompt": prompt,
                        "response": response,
                        "in_distribution": sample["in_distribution"],
                        "domain": sample["domain"],
                        "avg_entropy": avg_entropy,
                        "total_entropy": total_entropy,
                        "num_response_tokens": num_tokens,
                        # Convert to bits (base 2) for interpretability
                        "avg_entropy_bits": avg_entropy / np.log(2) if avg_entropy != 0 else 0.0,
                    })
                    
                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    results.append({
                        "model": save_name,
                        "sample_id": idx,
                        "prompt": prompt,
                        "response": response,
                        "in_distribution": sample.get("in_distribution"),
                        "domain": sample.get("domain"),
                        "avg_entropy": None,
                        "total_entropy": None,
                        "num_response_tokens": 0,
                        "avg_entropy_bits": None,
                    })
            
            # Clean up
            del model
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
    
    # Define your models
    model_names = [
        "allenai/Olmo-3-1025-7B",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_50",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_100",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_150",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_200",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_250",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_300",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_350",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_400",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_450",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_500",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_550",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_600",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_650",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_700",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_750",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_800",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_850",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_900",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_950",
        "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/baseline_random_all/test_exp__1__1761341180_checkpoints/step_1000",
    ]
    
    # Run evaluation
    results_df = evaluate_models_on_dataset(
        dataset=dataset,
        model_names=model_names,
        prompt_column="prompt",
        response_column="response"
    )
    
    # Save results
    results_df.to_csv("calc-likelihoods/entropy_results-100.csv", index=False)
    print("\n" + "="*80)
    print("Results saved to: calc-likelihoods/entropy_results-100.csv")
    print("="*80)
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY MODEL")
    print("="*80)
    summary = results_df.groupby('model').agg({
        'avg_entropy': ['mean', 'std', 'min', 'max'],
        'avg_entropy_bits': ['mean', 'std', 'min', 'max'],
        'total_entropy': ['mean', 'std'],
        'num_response_tokens': 'mean'
    }).round(4)
    print(summary)
    
    # Display individual results
    print("\n" + "="*80)
    print("DETAILED RESULTS")
    print("="*80)
    for model in results_df['model'].unique():
        print(f"\nModel: {model}")
        model_results = results_df[results_df['model'] == model]
        print(model_results[['sample_id', 'avg_entropy', 'avg_entropy_bits', 'total_entropy', 'num_response_tokens']])


if __name__ == "__main__":
    main()