#!/usr/bin/env python3
"""
Benchmark script for testing vLLM generator performance.

This script loads datasets in the same way as grpo_fast.py, sets up a generator
like in test_grpo_fast.py, and streams results to/from the generator to measure
performance.
"""

import argparse
import logging
import queue
import threading
import time
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import ray
import torch
import transformers
import vllm
import torch.utils.flop_counter

from open_instruct import dataset_transformation
from open_instruct import grpo_fast
from open_instruct import vllm_utils3
from open_instruct.utils import ArgumentParserPlus
from open_instruct.model_utils import ModelConfig
from open_instruct.dataset_transformation import TokenizerConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_flops_forward(model: torch.nn.Module, inp: torch.Tensor) -> int:
    """
    Calculates FLOPs for the forward pass of a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        inp (torch.Tensor or Tuple): The input tensor or a tuple of input tensors
                                     for the model.
    Returns:
        int: Total FLOPs for the forward pass.
    """
    istrain = model.training
    model.eval()  # Set model to evaluation mode for consistent FLOPs counting
    
    # Ensure input is a tensor
    if not isinstance(inp, torch.Tensor):
        # Assuming input is a shape tuple if not a tensor
        inp = torch.randn(inp) 

    flop_counter = torch.utils.flop_counter.FlopCounterMode(display=False, depth=None)
    with flop_counter:
        model(inp)
    
    total_flops = flop_counter.get_total_flops()

    if istrain:
        model.train() # Restore original training state
    return total_flops


def calculate_model_flops_per_token(model_config: Any, model_path: str) -> float:
    """
    Calculate actual FLOPs per token for a transformer model using torch FlopCounterMode.
    
    Args:
        model_config: HuggingFace model config
        model_path: Path to the actual model for precise measurement
        
    Returns:
        float: FLOPs per token
    """
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        trust_remote_code=True
    )
    
    # Create a single token input
    input_ids = torch.tensor([[1]], device=model.device)  # Single token
    
    flops = get_flops_forward(model, input_ids)
    
    # Clean up
    del model
    torch.cuda.empty_cache()
    
    return flops


def get_gpu_peak_flops() -> float:
    """
    Get theoretical peak FLOPs for the current GPU.
    
    Returns:
        float: Peak FLOPs per second
    """
    if not torch.cuda.is_available():
        return 0.0
        
    device_name = torch.cuda.get_device_name(0).lower().replace(' ', '').replace('-', '')
    
    # Peak FLOPs for common GPUs (bfloat16/float16)
    gpu_peak_flops = {
        'a100':  312e12,  # 312 TFLOPs for bf16
        'b200': 2250e12,  # 2250 TFLOPS for bf16. 
        'h100':  990e12,  # 990 TFLOPs for bf16
        'nvidiartxa6000': 150e12,
    }
    
    # Try to match GPU name
    for gpu_key, flops in gpu_peak_flops.items():
        if gpu_key.replace(' ', '').replace('-', '') in device_name:
            logger.info(f"Detected GPU: {device_name}, Peak FLOPs: {flops/1e12:.0f} TFLOPs")
            return flops
    raise ValueError(f"Unknown device: {device_name}.")


def calculate_mfu(tokens_per_second: float, model_flops_per_token: float, gpu_peak_flops: float) -> float:
    """
    Calculate Model FLOPs Utilization (MFU).
    
    Args:s
        tokens_per_second: Actual token generation rate
        model_flops_per_token: Theoretical FLOPs per token for the model
        gpu_peak_flops: Peak FLOPs of the GPU
        
    Returns:
        float: MFU as a percentage (0-100)
    """
    if gpu_peak_flops == 0:
        return 0.0
        
    actual_flops_per_second = tokens_per_second * model_flops_per_token
    mfu_percentage = (actual_flops_per_second / gpu_peak_flops) * 100
    
    return mfu_percentage


@dataclass
class BenchmarkArgs:
    """Benchmark-specific arguments."""
    
    # Dataset configuration
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["hamishivi/hamishivi_rlvr_orz_math_57k_collected_all_filtered_hamishivi_qwen2_5_openthoughts2", "1.0"])
    """A list of datasets (local or HF) to sample from."""
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    """The dataset splits to use for training"""
    max_token_length: int = 10240
    """The maximum token length to use for the dataset"""
    max_prompt_token_length: int = 2048
    """The maximum prompt token length to use for the dataset"""
    
    # Generation configuration
    temperature: float = 1.0
    """the sampling temperature"""
    max_tokens: int = 16384
    """Maximum tokens to generate (response_length)"""
    top_p: float = 0.9
    """Top-p for nucleus sampling"""
    
    # Benchmark configuration
    num_batches: int = 5
    """Number of batches to process"""
    batch_size: int = 256
    """Batch size for generation (num_unique_prompts_rollout)"""
    num_samples_per_prompt: int = 16
    """Number of samples per prompt (num_samples_per_prompt_rollout)"""
    
    # vLLM configuration
    num_engines: int = 8
    """Number of vLLM engines to use"""
    tensor_parallel_size: int = 1
    """Tensor parallel size for vLLM"""
    max_model_len: int = 20480
    """Maximum model sequence length"""
    vllm_gpu_memory_utilization: float = 0.9
    """GPU memory utilization for vLLM"""
    
    # Other configurations
    dataset_cache_mode: str = "local"
    """The mode to use for caching the dataset."""
    dataset_local_cache_dir: str = "benchmark_cache"
    """The directory to save the local dataset cache to."""
    dataset_skip_cache: bool = False
    """Whether to skip the cache."""
    dataset_transform_fn: List[str] = field(default_factory=lambda: ["rlvr_tokenize_v1", "rlvr_filter_v1"])
    """The list of transform functions to apply to the dataset."""
    seed: int = 42
    """Seed of the experiment"""


@dataclasses.dataclass
class BenchmarkConfig:
    """Configuration for the benchmark - created from parsed arguments."""
    
    # Model configuration
    model_name: str
    max_model_len: int
    vllm_gpu_memory_utilization: float
    
    # Dataset configuration
    dataset_mixer_list: List[str]
    dataset_mixer_list_splits: List[str]
    max_token_length: int
    max_prompt_token_length: int
    
    # Generation configuration
    temperature: float
    max_tokens: int
    top_p: float
    
    # Benchmark configuration
    num_batches: int
    batch_size: int
    num_samples_per_prompt: int
    
    # vLLM configuration 
    num_engines: int
    tensor_parallel_size: int
    
    # Chat template configuration
    chat_template_name: str
    add_bos: bool


def setup_tokenizer(config: BenchmarkConfig) -> Tuple[Any, Any, float, float]:
    """Set up the tokenizer and model config."""
    logger.info(f"Loading tokenizer and config: {config.model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load model config for FLOPs calculation
    model_config = transformers.AutoConfig.from_pretrained(config.model_name)
    model_flops_per_token = calculate_model_flops_per_token(model_config, config.model_name)
    gpu_peak_flops = get_gpu_peak_flops()
    
    logger.info(f"Model FLOPs per token: {model_flops_per_token/1e9:.2f} GFLOPs")
    logger.info(f"GPU peak FLOPs: {gpu_peak_flops/1e12:.0f} TFLOPs")
    
    return tokenizer, model_config, model_flops_per_token, gpu_peak_flops


def setup_dataset(config: BenchmarkConfig, benchmark_args: BenchmarkArgs) -> Any:
    """Set up the dataset using the same pipeline as grpo_fast.py."""
    logger.info("Loading and processing dataset...")
    
    # Create tokenizer config (matching long_repro_script.sh)
    tc = dataset_transformation.TokenizerConfig(
        tokenizer_name_or_path=config.model_name,
        trust_remote_code=True,
        chat_template_name=config.chat_template_name,
        add_bos=config.add_bos,
    )
    
    # Transform function arguments
    transform_fn_args = [
        {},  # For rlvr_tokenize_v1
        {
            "max_token_length": config.max_token_length,
            "max_prompt_token_length": config.max_prompt_token_length,
        },  # For rlvr_filter_v1
    ]
    
    # Load dataset
    dataset = dataset_transformation.get_cached_dataset_tulu(
        dataset_mixer_list=config.dataset_mixer_list,
        dataset_mixer_list_splits=config.dataset_mixer_list_splits,
        tc=tc,
        dataset_transform_fn=benchmark_args.dataset_transform_fn,
        transform_fn_args=transform_fn_args,
        dataset_cache_mode=benchmark_args.dataset_cache_mode,
        dataset_local_cache_dir=benchmark_args.dataset_local_cache_dir,
        dataset_skip_cache=benchmark_args.dataset_skip_cache,
    )
    
    # Shuffle dataset
    dataset = dataset.shuffle(seed=benchmark_args.seed)
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    return dataset


def setup_vllm_engines(config: BenchmarkConfig) -> List[Any]:
    """Set up vLLM engines."""
    logger.info("Setting up vLLM engines...")
    
    # Initialize Ray
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_cpus=4, num_gpus=1, ignore_reinit_error=True)
    
    # Create placement group for multiple engines
    bundles = [{"GPU": 1, "CPU": 1} for _ in range(config.num_engines)]
    pg = ray.util.placement_group(bundles, strategy="PACK")
    ray.get(pg.ready())
    
    # Create vLLM engines
    vllm_engines = vllm_utils3.create_vllm_engines(
        num_engines=config.num_engines,
        tensor_parallel_size=config.tensor_parallel_size,
        enforce_eager=True,
        tokenizer_name_or_path=config.model_name,
        pretrain=config.model_name,
        revision=None,
        seed=42,
        enable_prefix_caching=False,
        max_model_len=config.max_model_len,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
        single_gpu_mode=False,
        pg=pg,
        tools={},
        max_tool_calls=[0],
    )
    
    logger.info("vLLM engines ready")
    
    return vllm_engines


def get_batch_data(dataset: Any, config: BenchmarkConfig, batch_idx: int) -> Tuple[List[List[int]], List[str], List[str]]:
    """Get a batch of data from the dataset."""
    start_idx = batch_idx * config.batch_size
    end_idx = min(start_idx + config.batch_size, len(dataset))
    
    batch_data = dataset[start_idx:end_idx]
    
    # Extract prompts and ground truths
    prompts = batch_data[dataset_transformation.INPUT_IDS_PROMPT_KEY]
    ground_truths = batch_data[dataset_transformation.GROUND_TRUTHS_KEY]
    datasets_list = batch_data[dataset_transformation.DATASET_SOURCE_KEY]
    
    # Expand if multiple samples per prompt
    if config.num_samples_per_prompt > 1:
        prompts = [prompt for prompt in prompts for _ in range(config.num_samples_per_prompt)]
        ground_truths = [gt for gt in ground_truths for _ in range(config.num_samples_per_prompt)]
        datasets_list = [ds for ds in datasets_list for _ in range(config.num_samples_per_prompt)]
        
    return prompts, ground_truths, datasets_list


def run_generation_batch(vllm_engines: List[Any], config: BenchmarkConfig, model_flops_per_token: float, gpu_peak_flops: float, prompts: List[List[int]], batch_idx: int) -> Dict[str, Any]:
    """Run generation for a batch of prompts and measure performance."""
    
    # Create queues
    inference_results_Q = queue.Queue(maxsize=10)
    param_prompt_Q = queue.Queue(maxsize=10)
    evaluation_inference_results_Q = queue.Queue(maxsize=10)
    
    # Create sampling parameters
    generation_config = vllm.SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
    )
    
    eval_generation_config = vllm.SamplingParams(
        temperature=0.0,
        max_tokens=config.max_tokens,
    )
    
    # Start vLLM generation thread
    def wrapped_vllm_generate_thread() -> None:
        try:
            grpo_fast.vllm_generate_thread(
                vllm_engines,
                generation_config,
                eval_generation_config,
                inference_results_Q,
                param_prompt_Q,
                1,  # num_training_steps
                None,  # eval_prompt_token_ids
                evaluation_inference_results_Q,
                2,  # eval_freq
                1,  # resume_training_step
                False,  # tool_use
            )
        except Exception as e:
            logger.error(f"Error in vllm_generate_thread: {e}")
            inference_results_Q.put(("ERROR", str(e), [], []))
            raise
    
    thread = threading.Thread(target=wrapped_vllm_generate_thread)
    
    # Measure timing
    start_time = time.time()
    
    # Start thread and send prompts
    thread.start()
    param_prompt_Q.put((None, prompts))
    
    # Get results
    try:
        result = inference_results_Q.get(timeout=120)
        
        if result[0] == "ERROR":
            logger.error(f"Generation failed: {result[1]}")
            return {"error": result[1]}
            
        response_ids, finish_reasons, masks, info = result
        
        # Calculate timing and throughput
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Calculate tokens generated (response_ids only contains newly generated tokens)
        total_new_tokens = sum(len(response) for response in response_ids)
        total_prompt_tokens = sum(len(prompt) for prompt in prompts)
        total_tokens_generated = total_new_tokens + total_prompt_tokens
        
        tokens_per_second = total_new_tokens / generation_time if generation_time > 0 else 0
        total_tokens_per_second = total_tokens_generated / generation_time if generation_time > 0 else 0
        
        # Calculate MFU
        mfu_percentage = calculate_mfu(tokens_per_second, model_flops_per_token, gpu_peak_flops)
        
        # Send stop signal
        param_prompt_Q.put(None)
        
    finally:
        thread.join(timeout=10)
        
    return {
        "batch_idx": batch_idx,
        "batch_size": len(prompts),
        "generation_time": generation_time,
        "total_tokens_generated": total_tokens_generated,
        "total_new_tokens": total_new_tokens,
        "tokens_per_second": tokens_per_second,
        "total_tokens_per_second": total_tokens_per_second,
        "mfu_percentage": mfu_percentage,
        "avg_new_tokens_per_sample": total_new_tokens / len(prompts) if prompts else 0,
        "finish_reasons": finish_reasons,
        "response_lengths": [len(response) for response in response_ids],
    }


def run_benchmark(dataset: Any, vllm_engines: List[Any], config: BenchmarkConfig, model_flops_per_token: float, gpu_peak_flops: float) -> List[Dict[str, Any]]:
    """Run the full benchmark."""
    logger.info(f"Starting benchmark with {config.num_batches} batches of size {config.batch_size}")
    
    all_results = []
    total_start_time = time.time()
    
    for batch_idx in range(config.num_batches):
        logger.info(f"Processing batch {batch_idx + 1}/{config.num_batches}")
        
        # Get batch data
        prompts, ground_truths, datasets_list = get_batch_data(dataset, config, batch_idx)
        
        if not prompts:
            logger.warning(f"No prompts in batch {batch_idx}, skipping")
            continue
            
        # Run generation
        batch_result = run_generation_batch(vllm_engines, config, model_flops_per_token, gpu_peak_flops, prompts, batch_idx)
        
        if "error" not in batch_result:
            all_results.append(batch_result)
            logger.info(f"Batch {batch_idx + 1} completed: "
                      f"{batch_result['tokens_per_second']:.2f} new tokens/sec, "
                      f"MFU: {batch_result['mfu_percentage']:.2f}%, "
                      f"{batch_result['generation_time']:.2f}s")
        else:
            logger.error(f"Batch {batch_idx + 1} failed: {batch_result['error']}")
            
    total_time = time.time() - total_start_time
    
    # Calculate summary statistics
    if all_results:
        print_summary(all_results, total_time, config, model_flops_per_token, gpu_peak_flops)
    else:
        logger.error("No successful batches completed")
        
    return all_results


def print_summary(results: List[Dict[str, Any]], total_time: float, config: BenchmarkConfig, model_flops_per_token: float, gpu_peak_flops: float) -> None:
    """Print benchmark summary statistics."""
    
    # Calculate metrics for all batches
    total_samples = sum(r["batch_size"] for r in results)
    total_new_tokens = sum(r["total_new_tokens"] for r in results)
    total_tokens = sum(r["total_tokens_generated"] for r in results)
    total_generation_time = sum(r["generation_time"] for r in results)
    
    avg_new_tokens_per_second = total_new_tokens / total_generation_time if total_generation_time > 0 else 0
    avg_total_tokens_per_second = total_tokens / total_generation_time if total_generation_time > 0 else 0
    avg_generation_time = total_generation_time / len(results)
    
    # Calculate average MFU
    avg_mfu = sum(r["mfu_percentage"] for r in results) / len(results) if results else 0
    
    throughput_samples_per_second = total_samples / total_time
    
    # Calculate metrics excluding the first batch (N-1 batches)
    if len(results) > 1:
        last_n_minus_1_results = results[1:]  # Skip first batch
        last_n_minus_1_samples = sum(r["batch_size"] for r in last_n_minus_1_results)
        last_n_minus_1_new_tokens = sum(r["total_new_tokens"] for r in last_n_minus_1_results)
        last_n_minus_1_tokens = sum(r["total_tokens_generated"] for r in last_n_minus_1_results)
        last_n_minus_1_generation_time = sum(r["generation_time"] for r in last_n_minus_1_results)
        
        avg_new_tokens_per_second_last_n_minus_1 = last_n_minus_1_new_tokens / last_n_minus_1_generation_time if last_n_minus_1_generation_time > 0 else 0
        avg_total_tokens_per_second_last_n_minus_1 = last_n_minus_1_tokens / last_n_minus_1_generation_time if last_n_minus_1_generation_time > 0 else 0
        avg_generation_time_last_n_minus_1 = last_n_minus_1_generation_time / len(last_n_minus_1_results)
        avg_mfu_last_n_minus_1 = sum(r["mfu_percentage"] for r in last_n_minus_1_results) / len(last_n_minus_1_results)
        
        # Calculate % wasted time due to variable generation lengths
        # This is the difference between time spent on longest vs shortest generations
        response_lengths_last_n_minus_1 = []
        for r in last_n_minus_1_results:
            response_lengths_last_n_minus_1.extend(r["response_lengths"])
        
        if response_lengths_last_n_minus_1:
            max_response_length = max(response_lengths_last_n_minus_1)
            min_response_length = min(response_lengths_last_n_minus_1)
            avg_response_length = sum(response_lengths_last_n_minus_1) / len(response_lengths_last_n_minus_1)
            
            # Estimate wasted time as the percentage of time spent on padding to max length
            # This is a rough estimate: (max_length - avg_length) / max_length
            wasted_time_percentage = ((max_response_length - avg_response_length) / max_response_length * 100) if max_response_length > 0 else 0
        else:
            wasted_time_percentage = 0
    else:
        # If only one batch, use the same metrics
        avg_new_tokens_per_second_last_n_minus_1 = avg_new_tokens_per_second
        avg_total_tokens_per_second_last_n_minus_1 = avg_total_tokens_per_second
        avg_generation_time_last_n_minus_1 = avg_generation_time
        avg_mfu_last_n_minus_1 = avg_mfu
        wasted_time_percentage = 0
    
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Model: {config.model_name}")
    print(f"Total batches: {len(results)}")
    print(f"Total samples: {total_samples}")
    print(f"Batch size: {config.batch_size}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Temperature: {config.temperature}")
    print("-"*60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total generation time: {total_generation_time:.2f}s")
    print(f"Total new tokens generated: {total_new_tokens}")
    print(f"Total tokens processed: {total_tokens}")
    print("-"*60)
    if len(results) > 1:
        print("LAST N-1 BATCHES (excluding first batch):")
        print(f"Average new tokens/second: {avg_new_tokens_per_second_last_n_minus_1:.2f}")
        print(f"Average total tokens/second: {avg_total_tokens_per_second_last_n_minus_1:.2f}")
        print(f"Average MFU: {avg_mfu_last_n_minus_1:.2f}%")
        print(f"Average generation time per batch: {avg_generation_time_last_n_minus_1:.2f}s")
        print(f"Wasted time (variable lengths): {wasted_time_percentage:.2f}%")
        print(f"Average new tokens per sample: {last_n_minus_1_new_tokens / last_n_minus_1_samples:.2f}")
    else:
        print("RESULTS:")
        print(f"Average new tokens/second: {avg_new_tokens_per_second:.2f}")
        print(f"Average total tokens/second: {avg_total_tokens_per_second:.2f}")
        print(f"Average MFU: {avg_mfu:.2f}%")
        print(f"Average generation time per batch: {avg_generation_time:.2f}s")
        print(f"Throughput (samples/second): {throughput_samples_per_second:.2f}")
        print(f"Average new tokens per sample: {total_new_tokens / total_samples:.2f}")
    
    print("-"*60)
    print(f"Model FLOPs per token: {model_flops_per_token/1e9:.2f} GFLOPs")
    print(f"GPU peak FLOPs: {gpu_peak_flops/1e12:.0f} TFLOPs")
    print("="*60)


def cleanup(vllm_engines: Optional[List[Any]]) -> None:
    """Clean up resources."""
    if vllm_engines:
        for engine in vllm_engines:
            ray.kill(engine)
    if ray.is_initialized():
        ray.shutdown()


def main() -> None:
    """Main benchmark function."""
    # Parse arguments using ArgumentParserPlus
    parser = ArgumentParserPlus((BenchmarkArgs, TokenizerConfig, ModelConfig))
    benchmark_args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    
    # Override model defaults for benchmark
    if model_config.model_name_or_path is None:
        model_config.model_name_or_path = "hamishivi/qwen2_5_openthoughts2"
    
    # Override tokenizer defaults for benchmark
    if tokenizer_config.tokenizer_name_or_path is None:
        tokenizer_config.tokenizer_name_or_path = model_config.model_name_or_path
    # Always use tulu_thinker for benchmark if not specified via args
    if tokenizer_config.chat_template_name == "tulu":  # default value
        tokenizer_config.chat_template_name = "tulu_thinker"
    
    # Create configuration from parsed arguments
    config = BenchmarkConfig(
        model_name=model_config.model_name_or_path,
        max_model_len=benchmark_args.max_model_len,
        vllm_gpu_memory_utilization=benchmark_args.vllm_gpu_memory_utilization,
        temperature=benchmark_args.temperature,
        max_tokens=benchmark_args.max_tokens,
        top_p=benchmark_args.top_p,
        num_batches=benchmark_args.num_batches,
        batch_size=benchmark_args.batch_size,
        num_samples_per_prompt=benchmark_args.num_samples_per_prompt,
        dataset_mixer_list=benchmark_args.dataset_mixer_list,
        dataset_mixer_list_splits=benchmark_args.dataset_mixer_list_splits,
        max_token_length=benchmark_args.max_token_length,
        max_prompt_token_length=benchmark_args.max_prompt_token_length,
        chat_template_name=tokenizer_config.chat_template_name,
        add_bos=tokenizer_config.add_bos,
        num_engines=benchmark_args.num_engines,
        tensor_parallel_size=benchmark_args.tensor_parallel_size,
    )
    
    # Print configuration values for verification
    print("\n" + "="*60)
    print("BENCHMARK CONFIGURATION VALUES")
    print("="*60)
    print(f"Model name: {config.model_name}")
    print(f"Max model len: {config.max_model_len}")
    print(f"vLLM GPU memory utilization: {config.vllm_gpu_memory_utilization}")
    print(f"Temperature: {config.temperature}")
    print(f"Max tokens: {config.max_tokens}")
    print(f"Top-p: {config.top_p}")
    print(f"Num batches: {config.num_batches}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num samples per prompt: {config.num_samples_per_prompt}")
    print(f"Dataset mixer list: {config.dataset_mixer_list}")
    print(f"Dataset mixer list splits: {config.dataset_mixer_list_splits}")
    print(f"Max token length: {config.max_token_length}")
    print(f"Max prompt token length: {config.max_prompt_token_length}")
    print(f"Chat template name: {config.chat_template_name}")
    print(f"Add BOS: {config.add_bos}")
    print(f"Num engines: {config.num_engines}")
    print(f"Tensor parallel size: {config.tensor_parallel_size}")
    print("="*60 + "\n")
    
    # Run benchmark
    vllm_engines = None
    
    try:
        tokenizer, model_config_obj, model_flops_per_token, gpu_peak_flops = setup_tokenizer(config)
        dataset = setup_dataset(config, benchmark_args)
        vllm_engines = setup_vllm_engines(config)
        run_benchmark(dataset, vllm_engines, config, model_flops_per_token, gpu_peak_flops)
    finally:
        cleanup(vllm_engines)


if __name__ == "__main__":
    main()
