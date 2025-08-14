#!/usr/bin/env python3
"""
Local GRPO runner - runs GRPO fast implementation locally without Ray.
Directly instantiates LLMRayActor and reports token counts instead of learning.
"""

import logging
import os
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import List, Literal, Optional

import numpy as np
import torch
import vllm
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from open_instruct.dataset_transformation import (
    prepare_datasets,
)
from open_instruct.model_utils import (
    DatasetConfig,
    ModelConfig,
    TokenizerConfig,
    make_tokenizer,
)
from open_instruct.queue_types import GenerationResult, PromptRequest
from open_instruct.utils import (
    ArgumentParserPlus,
    get_torch_device_string,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class Args:
    # Data parameters
    dataset_mixer: Optional[dict[str, float]] = None
    dataset_mixer_dict: List[DatasetConfig] = field(default_factory=list)
    dataset_splits: Optional[List[str]] = None
    dataset_splits_list: List[str] = field(default_factory=list)
    
    # Generation parameters
    num_unique_prompts_rollout: int = 128
    num_samples_per_prompt_rollout: int = 2
    max_response_length: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stop_token_ids: Optional[List[int]] = None
    
    # VLLM parameters
    vllm_tensor_parallel_size: int = 1
    vllm_enforce_eager: bool = False
    vllm_gpu_memory_utilization: float = 0.8
    vllm_enable_prefix_caching: bool = False
    
    # Other parameters
    seed: int = 42
    num_iterations: int = 10  # Number of iterations to run
    batch_size: int = 32  # Batch size for generation
    
    # Model parameters
    chat_template_name: str = "simple_concat_with_space"
    

class LocalLLMEngine:
    """Local wrapper around vLLM engine without Ray."""
    
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        tensor_parallel_size: int = 1,
        enforce_eager: bool = False,
        gpu_memory_utilization: float = 0.8,
        enable_prefix_caching: bool = False,
        max_model_len: int = 4096,
        seed: int = 42,
    ):
        logger.info(f"Initializing local vLLM engine for {model_name_or_path}")
        
        # Create vLLM engine arguments
        engine_args = vllm.EngineArgs(
            model=model_name_or_path,
            tokenizer=tokenizer_name_or_path,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            max_model_len=max_model_len,
            seed=seed,
            distributed_executor_backend="mp" if tensor_parallel_size > 1 else None,
        )
        
        # Create the LLM engine
        self.llm_engine = vllm.LLMEngine.from_engine_args(engine_args)
        self.tokenizer = self.llm_engine.get_tokenizer()
        logger.info("vLLM engine initialized successfully")
    
    def generate(self, prompts: List[List[int]], sampling_params: vllm.SamplingParams) -> GenerationResult:
        """Generate responses for the given prompts."""
        # Add requests to the engine
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_id = f"req_{i}_{time.time()}"
            self.llm_engine.add_request(request_id, prompt, sampling_params)
            request_ids.append(request_id)
        
        # Process until all requests are complete
        outputs = {}
        while self.llm_engine.has_unfinished_requests():
            step_outputs = list(self.llm_engine.step())
            for output in step_outputs:
                if output.finished:
                    outputs[output.request_id] = output
        
        # Collect results in order
        responses = []
        finish_reasons = []
        masks = []
        
        for request_id in request_ids:
            output = outputs[request_id]
            # Get the best completion (assuming n=1)
            completion = output.outputs[0]
            
            # Extract token IDs
            token_ids = completion.token_ids
            responses.append(token_ids)
            
            # Finish reason
            finish_reasons.append(completion.finish_reason)
            
            # Create mask (all ones for now)
            masks.append([1] * len(token_ids))
        
        # Create dummy request info
        from open_instruct.queue_types import RequestInfo
        request_info = RequestInfo(
            num_calls=[0] * len(responses),
            timeouts=[0] * len(responses),
            tool_errors=[""] * len(responses),
            tool_outputs=[""] * len(responses),
            tool_runtimes=[0.0] * len(responses),
            tool_calleds=[False] * len(responses),
        )
        
        return GenerationResult(
            responses=responses,
            finish_reasons=finish_reasons,
            masks=masks,
            request_info=request_info,
        )


def setup_datasets(args: Args, tc: TokenizerConfig, tokenizer: PreTrainedTokenizer):
    """Setup training and eval datasets."""
    logger.info("Setting up datasets...")
    
    # Prepare dataset configuration
    dataset_config = {
        "dataset_mixer": args.dataset_mixer,
        "dataset_mixer_list": args.dataset_mixer_dict,
        "dataset_splits": args.dataset_splits,
        "dataset_splits_list": args.dataset_splits_list,
    }
    
    # Load datasets
    train_dataset, eval_dataset = prepare_datasets(
        dataset_mixer=dataset_config.get("dataset_mixer"),
        dataset_mixer_list=dataset_config.get("dataset_mixer_list"),
        dataset_splits=dataset_config.get("dataset_splits"),
        dataset_splits_list=dataset_config.get("dataset_splits_list"),
        chat_template=tc.chat_template_name,
        n_epochs=1,
        num_workers=4,
        add_bos=getattr(tc, "add_bos", False),
        seed=args.seed,
    )
    
    logger.info(f"Train dataset size: {len(train_dataset) if train_dataset else 0}")
    logger.info(f"Eval dataset size: {len(eval_dataset) if eval_dataset else 0}")
    
    return train_dataset, eval_dataset


def create_generation_config(args: Args, tokenizer: PreTrainedTokenizer) -> vllm.SamplingParams:
    """Create vLLM sampling parameters."""
    stop_token_ids = args.stop_token_ids or []
    if tokenizer.eos_token_id and tokenizer.eos_token_id not in stop_token_ids:
        stop_token_ids.append(tokenizer.eos_token_id)
    
    return vllm.SamplingParams(
        n=args.num_samples_per_prompt_rollout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length,
        stop_token_ids=stop_token_ids,
        include_stop_str_in_output=False,
    )


def main(args: Args, tc: TokenizerConfig, model_config: ModelConfig):
    """Main function to run local GRPO generation and token counting."""
    
    # Setup tokenizer
    tokenizer = make_tokenizer(tc, model_config)
    logger.info(f"Tokenizer loaded: {tc.tokenizer_name_or_path}")
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(args, tc, tokenizer)
    
    if not train_dataset:
        logger.error("No training dataset available!")
        return
    
    # Calculate max model length
    max_model_len = tc.max_seq_length or 4096
    max_prompt_len = max_model_len - args.max_response_length
    logger.info(f"Max model length: {max_model_len}, Max prompt length: {max_prompt_len}")
    
    # Initialize local vLLM engine
    engine = LocalLLMEngine(
        model_name_or_path=model_config.model_name_or_path,
        tokenizer_name_or_path=tc.tokenizer_name_or_path,
        tensor_parallel_size=args.vllm_tensor_parallel_size,
        enforce_eager=args.vllm_enforce_eager,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        enable_prefix_caching=args.vllm_enable_prefix_caching,
        max_model_len=max_model_len,
        seed=args.seed,
    )
    
    # Create generation config
    generation_config = create_generation_config(args, tokenizer)
    
    # Prepare data indices
    train_dataset_idxs = np.arange(len(train_dataset))
    
    # Token counting
    total_tokens_generated = 0
    total_prompts_processed = 0
    
    # Main generation loop with tqdm
    pbar = tqdm(total=args.num_iterations, desc="Generation iterations")
    
    for iteration in range(args.num_iterations):
        # Sample batch of prompts
        batch_indices = np.random.choice(
            train_dataset_idxs, 
            size=min(args.batch_size, len(train_dataset)),
            replace=False
        )
        
        # Get prompts from dataset
        prompts = []
        for idx in batch_indices:
            item = train_dataset[int(idx)]
            # Tokenize the prompt/query
            if isinstance(item, dict):
                if "query" in item:
                    prompt_text = item["query"]
                elif "prompt" in item:
                    prompt_text = item["prompt"]
                else:
                    prompt_text = str(item)
            else:
                prompt_text = str(item)
            
            # Tokenize
            prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
            
            # Truncate if needed
            if len(prompt_tokens) > max_prompt_len:
                prompt_tokens = prompt_tokens[:max_prompt_len]
            
            prompts.append(prompt_tokens)
        
        # Generate responses
        logger.info(f"Iteration {iteration + 1}: Generating for {len(prompts)} prompts")
        start_time = time.time()
        
        result = engine.generate(prompts, generation_config)
        
        generation_time = time.time() - start_time
        
        # Count tokens
        batch_tokens = sum(len(response) for response in result.responses)
        total_tokens_generated += batch_tokens
        total_prompts_processed += len(prompts)
        
        # Update progress bar
        tokens_per_sec = batch_tokens / generation_time if generation_time > 0 else 0
        pbar.set_postfix({
            "Total tokens": total_tokens_generated,
            "Batch tokens": batch_tokens,
            "Tokens/sec": f"{tokens_per_sec:.1f}",
            "Prompts": total_prompts_processed,
        })
        pbar.update(1)
        
        # Log some statistics
        if (iteration + 1) % 5 == 0:
            avg_response_len = batch_tokens / len(result.responses)
            logger.info(
                f"Stats - Iteration: {iteration + 1}, "
                f"Total tokens: {total_tokens_generated}, "
                f"Avg response length: {avg_response_len:.1f}, "
                f"Generation time: {generation_time:.2f}s"
            )
    
    pbar.close()
    
    # Final statistics
    logger.info("=" * 50)
    logger.info("Final Statistics:")
    logger.info(f"Total iterations: {args.num_iterations}")
    logger.info(f"Total prompts processed: {total_prompts_processed}")
    logger.info(f"Total tokens generated: {total_tokens_generated}")
    logger.info(f"Average tokens per prompt: {total_tokens_generated / total_prompts_processed:.1f}")
    logger.info("=" * 50)


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParserPlus((Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    
    # Validate types
    assert isinstance(args, Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)
    
    # Run main
    main(args, tokenizer_config, model_config)