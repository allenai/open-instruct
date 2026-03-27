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


CHECKPOINT_PREFIX = "/weka/oe-adapt-default/jacobm/social-rl/checkpoints/7b-instruct-sft-random-small/test_exp_small__1__1774136278_checkpoints/"


def evaluate_models_on_dataset(
    dataset: Dataset,
    model_names: List[str],
    prompt_column: str = "prompt",
    response_column: str = "response",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 1
) -> pd.DataFrame:
    """
    Evaluate multiple models on a dataset and calculate response log likelihoods.
    """
    results = []

    # Load tokenizer from one of the narrow-D checkpoints
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PREFIX + "step_800/")

    for model_name in model_names:
        if "Olmo" in model_name:
            save_name = "step_0"
        else:
            save_name = model_name.replace(CHECKPOINT_PREFIX, "").strip("/")
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
            for idx, sample in enumerate(tqdm(dataset, desc=f"Processing {save_name}")):
                messages = sample["messages"]
                prompt = messages[0]["content"]
                response = messages[1]["content"]

                try:
                    avg_log_likelihood, num_tokens = calculate_response_log_likelihood(
                        model, tokenizer, prompt, response, device
                    )

                    results.append({
                        "model": save_name,
                        "sample_id": idx,
                        "prompt": prompt,
                        "response": response,
                        "in_distribution": sample["in_distribution"],
                        "domain": sample["domain"],
                        "avg_log_likelihood": avg_log_likelihood,
                        "num_response_tokens": num_tokens,
                        "perplexity": np.exp(-avg_log_likelihood) if avg_log_likelihood != 0 else float('inf')
                    })

                except Exception as e:
                    print(f"Error processing sample {idx}: {e}")
                    results.append({
                        "model": save_name,
                        "sample_id": idx,
                        "prompt": prompt,
                        "response": response,
                        "avg_log_likelihood": None,
                        "num_response_tokens": 0,
                        "perplexity": None
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
    # Use SFT eval dataset since narrow-D was trained on top of SFT
    dataset = load_data_from_huggingface("jacobmorrison/social-rl-eval-dataset-SFT-100", split="train")

    # Narrow-D checkpoints (100 math prompts, SFT-based)
    model_names = [
        "jacobmorrison/Olmo-3-7B-Instruct-SFT-do-sample",
        CHECKPOINT_PREFIX + "step_50",
        CHECKPOINT_PREFIX + "step_100",
        CHECKPOINT_PREFIX + "step_150",
        CHECKPOINT_PREFIX + "step_200",
        CHECKPOINT_PREFIX + "step_250",
        CHECKPOINT_PREFIX + "step_300",
        CHECKPOINT_PREFIX + "step_350",
        CHECKPOINT_PREFIX + "step_400",
        CHECKPOINT_PREFIX + "step_450",
        CHECKPOINT_PREFIX + "step_500",
        CHECKPOINT_PREFIX + "step_550",
        CHECKPOINT_PREFIX + "step_600",
        CHECKPOINT_PREFIX + "step_650",
        CHECKPOINT_PREFIX + "step_700",
        CHECKPOINT_PREFIX + "step_750",
        CHECKPOINT_PREFIX + "step_800",
    ]

    # Run evaluation
    results_df = evaluate_models_on_dataset(
        dataset=dataset,
        model_names=model_names,
        prompt_column="prompt",
        response_column="response"
    )

    # Save results
    results_df.to_csv("calc-likelihoods/log_likelihood_results-SFT-small-100.csv", index=False)
    print("\n" + "="*80)
    print("Results saved to: calc-likelihoods/log_likelihood_results-SFT-small-100.csv")
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
    for model in results_df['model'].unique():
        print(f"\nModel: {model}")
        model_results = results_df[results_df['model'] == model]
        print(model_results[['sample_id', 'avg_log_likelihood', 'perplexity', 'num_response_tokens']])


if __name__ == "__main__":
    main()
