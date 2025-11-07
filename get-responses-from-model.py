"""
Efficient batch inference script using vLLM for HuggingFace datasets.
Supports 1-2 GPUs and generates multiple responses per prompt.
"""

import argparse
import torch
from datasets import load_dataset, Dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict, Union
import json
from tqdm import tqdm


def setup_vllm_model(
    model_name: str,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096
) -> LLM:
    """
    Initialize vLLM model with appropriate settings.
    
    Args:
        model_name: HuggingFace model name or path
        tensor_parallel_size: Number of GPUs to use (1 or 2)
        gpu_memory_utilization: Fraction of GPU memory to use
        max_model_len: Maximum sequence length
    """
    print(f"Loading model: {model_name}")
    print(f"Using {tensor_parallel_size} GPU(s)")
    
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True
    )
    
    return llm


def create_sampling_params(
    n: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 512,
    **kwargs
) -> SamplingParams:
    """
    Create sampling parameters for generation.
    
    Args:
        n: Number of responses to generate per prompt
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens to generate
    """
    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        **kwargs
    )


def apply_chat_template(
    prompts: Union[List[str], List[List[Dict]]],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = True
) -> List[str]:
    """
    Apply chat template to prompts.
    
    Args:
        prompts: Either list of strings or list of message dictionaries
        tokenizer: Tokenizer with chat template
        add_generation_prompt: Whether to add generation prompt
    
    Returns:
        List of formatted prompts
    """
    formatted_prompts = []
    
    for prompt in tqdm(prompts, desc="Applying chat template"):
        if isinstance(prompt, str):
            # Convert string to message format
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            # Already in message format
            messages = prompt
        else:
            raise ValueError(f"Unsupported prompt type: {type(prompt)}")
        
        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        formatted_prompts.append(formatted)
    
    return formatted_prompts


def batch_generate(
    llm: LLM,
    prompts: List[str],
    sampling_params: SamplingParams,
    batch_size: int = 1000
) -> List[List[str]]:
    """
    Generate responses for all prompts in batches.
    
    Args:
        llm: vLLM model instance
        prompts: List of input prompts
        sampling_params: Sampling parameters
        batch_size: Number of prompts to process at once (vLLM handles this internally)
    
    Returns:
        List of lists, where each inner list contains n responses for that prompt
    """
    print(f"Generating {sampling_params.n} response(s) per prompt for {len(prompts)} prompts...")
    
    # vLLM handles batching internally, but we can process in chunks for progress tracking
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Processing batches"):
        batch_prompts = prompts[i:i + batch_size]
        
        # Generate responses
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Extract responses for each prompt
        for output in outputs:
            responses = [out.text for out in output.outputs]
            all_responses.append(responses)
    
    return all_responses


def process_dataset(
    dataset_name: str,
    model_name: str,
    tokenizer_name: str,
    output_path: str,
    num_gpus: int = 1,
    num_responses: int = 1,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 512,
    prompt_column: str = "prompt",
    split: str = "train",
    subset: str = None,
    max_samples: int = None,
    gpu_memory_utilization: float = 0.9,
    max_model_len: int = 4096,
    apply_chat_template_flag: bool = False,
    messages_column: str = None
):
    """
    Main processing function.
    
    Args:
        dataset_name: HuggingFace dataset name or path
        model_name: Model name or path
        output_path: Path to save results
        num_gpus: Number of GPUs to use (1 or 2)
        num_responses: Number of responses per prompt (1-5)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        max_tokens: Maximum tokens per response
        prompt_column: Name of the column containing prompts
        split: Dataset split to use
        subset: Dataset subset/configuration
        max_samples: Limit number of samples (for testing)
        gpu_memory_utilization: GPU memory utilization fraction
        max_model_len: Maximum model sequence length
        apply_chat_template_flag: Whether to apply tokenizer's chat template
        messages_column: Column name for messages (if different from prompt_column)
    """
    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    if subset:
        dataset = load_dataset(dataset_name, subset, split=split)
    else:
        dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"Dataset size: {len(dataset)}")
    
    # Extract prompts
    column_to_use = messages_column if messages_column else prompt_column
    if column_to_use not in dataset.column_names:
        raise ValueError(f"Column '{column_to_use}' not found. Available columns: {dataset.column_names}")
    
    prompts = dataset[column_to_use]
    
    # Apply chat template if requested
    if apply_chat_template_flag:
        print("Loading tokenizer for chat template...")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        
        if not hasattr(tokenizer, 'chat_template') or tokenizer.chat_template is None:
            print("Warning: Model does not have a chat template. Proceeding without template.")
        else:
            print("Applying chat template to prompts...")
            prompts = apply_chat_template(prompts, tokenizer)
    
    # Setup model
    llm = setup_vllm_model(
        model_name=model_name,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len
    )
    
    # Setup sampling parameters
    sampling_params = create_sampling_params(
        n=num_responses,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Generate responses
    all_responses = batch_generate(llm, prompts, sampling_params)
    
    # Add responses to dataset
    print("Mapping responses back to dataset...")
    
    # Add metadata
    new_columns = {}
    
    if num_responses == 1:
        # Single response: add as single column
        new_columns['generated_response'] = [responses[0] for responses in all_responses]
    else:
        # Multiple responses: add numbered columns
        for i in range(num_responses):
            new_columns[f'generated_response_{i+1}'] = [
                responses[i] if i < len(responses) else "" 
                for responses in all_responses
            ]
    
    # Add generation metadata
    new_columns['num_responses_generated'] = [len(responses) for responses in all_responses]
    new_columns['model_name'] = [model_name] * len(dataset)
    new_columns['temperature'] = [temperature] * len(dataset)
    new_columns['top_p'] = [top_p] * len(dataset)
    new_columns['max_tokens'] = [max_tokens] * len(dataset)
    
    # Create new dataset with original + generated columns
    result_dataset = dataset.add_column('generated_responses_list', all_responses)
    for col_name, col_data in new_columns.items():
        result_dataset = result_dataset.add_column(col_name, col_data)
    
    # Save dataset
    print(f"Saving results to {output_path}")
    
    if output_path.endswith('.jsonl'):
        result_dataset.to_json(output_path, lines=True)
    elif output_path.endswith('.parquet'):
        result_dataset.to_parquet(output_path)
    elif output_path.endswith('.csv'):
        result_dataset.to_csv(output_path)
    else:
        # Default to parquet
        result_dataset.save_to_disk(output_path)
    
    print("Done!")
    print(f"Processed {len(result_dataset)} examples")
    print(f"Generated {num_responses} response(s) per prompt")


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with vLLM on HuggingFace datasets"
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name or local path"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        required=True,
        help="Tokenizer name or path (e.g., 'mistralai/Mistral-7B-Instruct-v0.2')"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path (.jsonl, .parquet, .csv, or directory)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        choices=[1, 2],
        help="Number of GPUs to use (1 or 2)"
    )
    parser.add_argument(
        "--num_responses",
        type=int,
        default=1,
        help="Number of responses per prompt (1-5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum tokens per response"
    )
    parser.add_argument(
        "--prompt_column",
        type=str,
        default="prompt",
        help="Name of column containing prompts"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/configuration"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.0-1.0)"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply tokenizer's chat template to prompts"
    )
    parser.add_argument(
        "--messages_column",
        type=str,
        default=None,
        help="Column name for messages (if using chat template with messages format)"
    )
    
    args = parser.parse_args()
    
    # Validate num_responses
    if args.num_responses < 1 or args.num_responses > 5:
        raise ValueError("num_responses must be between 1 and 5")
    
    print(args)
    
    # Process dataset
    process_dataset(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        tokenizer_name=args.tokenizer_name,
        output_path=args.output_path,
        num_gpus=args.num_gpus,
        num_responses=args.num_responses,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        prompt_column=args.prompt_column,
        split=args.split,
        subset=args.subset,
        max_samples=args.max_samples,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        apply_chat_template_flag=args.apply_chat_template,
        messages_column=args.messages_column
    )


if __name__ == "__main__":
    main()