#!/usr/bin/env python3
"""
Local GRPO runner - runs GRPO fast implementation locally without Ray for LLMRayActor.
Directly instantiates LLMRayActor locally and reports token counts instead of learning.
"""

import logging
import os

# Set NCCL_CUMEM_ENABLE for performance reasons before importing vllm
os.environ["NCCL_CUMEM_ENABLE"] = "0"
import threading
import time
from concurrent import futures
from queue import Empty, Queue
from typing import List, Optional

import numpy as np
import ray
from ray.util import queue as ray_queue
from tqdm import tqdm

# Import grpo_fast as a module
import open_instruct.grpo_fast as grpo_fast

# Import from other modules
from open_instruct.dataset_transformation import TokenizerConfig
from open_instruct.model_utils import ModelConfig
from open_instruct.rl_utils2 import Timer
from open_instruct.utils import ArgumentParserPlus
from open_instruct.vllm_utils3 import ActorManager, LLMRayActor


class MockQueue:
    """Mock queue that works like a Ray queue but uses Python Queue internally."""
    def __init__(self, maxsize=0):
        self._queue = Queue(maxsize=maxsize)
    
    def put(self, item, timeout=None):
        """Put item in queue, compatible with Ray queue interface."""
        self._queue.put(item, timeout=timeout)
    
    def get(self, timeout=None):
        """Get item from queue, compatible with Ray queue interface."""
        return self._queue.get(timeout=timeout)
    
    def empty(self):
        """Check if queue is empty."""
        return self._queue.empty()
    
    def qsize(self):
        """Get queue size."""
        return self._queue.qsize()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def local_generate_thread(vllm_actors, stop_event):
    """Thread function that repeatedly calls process_from_queue on local vLLM actors."""
    logger.info("[Generate Thread] 🚀 Starting local generation thread")
    while not stop_event.is_set():
        with Timer("🔥 Generation time") as timer:
            # Use list comprehension like grpo_fast.py
            processed_results = [actor.process_from_queue(timeout=20) for actor in vllm_actors]
            num_processed = sum(int(result) for result in processed_results)
            # Suppress timing output if nothing was processed
            if num_processed == 0:
                timer.noop = True
    logger.info("[Generate Thread] 🛑 Stopping generation thread")


def main(args: grpo_fast.Args, tc: TokenizerConfig, model_config: ModelConfig):
    """Main function to run local GRPO generation and token counting."""
    
    # Setup tokenizer (needed for padding)
    tokenizer = grpo_fast.make_tokenizer(tc, model_config)
    logger.info(f"Tokenizer loaded: {tc.tokenizer_name_or_path}")
    
    # Setup tools (matching grpo_fast.py)
    tool_objects = {}
    if args.tools:
        logger.info(f"Setting up tools: {args.tools}")
        for tool in args.tools:
            if tool.lower() == "search":
                from open_instruct.search_utils.search_tool import SearchTool
                tool = SearchTool(
                    start_str="<query>",
                    end_str="</query>",
                    api_endpoint=args.search_api_endpoint,
                    number_documents_to_search=args.number_documents_to_search,
                )
                tool_objects[tool.end_str] = tool
                args.stop_strings.append(tool.end_str)
                logger.info(f"Added search tool with end_str: {tool.end_str}")
            elif tool.lower() == "code":
                from open_instruct.tool_utils.tool_vllm import PythonCodeTool
                tool = PythonCodeTool(
                    start_str="<code>", 
                    end_str="</code>", 
                    api_endpoint=args.code_tool_api_endpoint
                )
                tool_objects[tool.end_str] = tool
                args.stop_strings.append(tool.end_str)
                logger.info(f"Added code tool with end_str: {tool.end_str}")
        logger.info(f"Tool objects created: {list(tool_objects.keys())}")
        logger.info(f"Updated stop_strings: {args.stop_strings}")
    else:
        logger.info("No tools configured")
    
    # Setup runtime variables
    args.run_name = f"{args.exp_name}_local__{args.seed}__{int(time.time())}"
    args.world_size = 1  # Single GPU for local
    args.num_training_steps = args.num_training_steps or 10
    
    # Setup datasets
    train_dataset, eval_dataset = grpo_fast.setup_datasets(args, tc, tokenizer)
    
    if not train_dataset:
        logger.error("No training dataset available!")
        return
    
    # Calculate max model length
    max_model_len = args.max_token_length
    
    # Initialize Ray before creating Ray objects (same as grpo_fast.py)
    ray.init(dashboard_host="0.0.0.0")
    
    # Create Ray queues for vLLM communication
    queue_size = (args.async_steps + 1) * args.vllm_num_engines
    inference_results_Q = ray_queue.Queue(maxsize=queue_size)
    param_prompt_Q = ray_queue.Queue(maxsize=queue_size)
    evaluation_inference_results_Q = ray_queue.Queue(maxsize=args.vllm_num_engines)
    
    # Create Python queue for packed sequences
    packed_sequences_Q = Queue(maxsize=args.async_steps)
    
    # Create ActorManager as a Ray actor
    actor_manager = ActorManager.remote()
    
    # Create generation configs
    generation_configs = grpo_fast.create_generation_configs(args)
    
    # Convert max_tool_calls to a dict mapping tool end strings to their limits
    if tool_objects:
        assert len(args.max_tool_calls) == 1 or len(args.max_tool_calls) == len(tool_objects), (
            "max_tool_calls must have length 1 (applies to all tools) or same length as tools (per-tool limit)"
        )
        if len(args.max_tool_calls) == 1:
            max_tool_calls_dict = {end_str: args.max_tool_calls[0] for end_str in tool_objects.keys()}
        else:
            max_tool_calls_dict = {end_str: limit for end_str, limit in zip(tool_objects.keys(), args.max_tool_calls)}
    else:
        max_tool_calls_dict = {}
    
    # Calculate inference_batch_size (same as grpo_fast.py)
    if args.inference_batch_size is None:
        args.inference_batch_size = args.num_unique_prompts_rollout // args.vllm_num_engines
        logger.info(
            f"Setting inference_batch_size to {args.inference_batch_size} "
            f"(num_unique_prompts_rollout={args.num_unique_prompts_rollout} // vllm_num_engines={args.vllm_num_engines})"
        )
    
    # Initialize local LLMRayActor instances (not as ray.remote)
    logger.info(f"Initializing {args.vllm_num_engines} local LLMRayActor instances...")
    vllm_actors = []
    for i in range(args.vllm_num_engines):
        logger.info(f"Creating LLMRayActor {i}...")
        llm_actor = LLMRayActor(
            model_config.model_name_or_path,
            revision=model_config.model_revision,
            tokenizer=tc.tokenizer_name_or_path,
            tokenizer_revision=model_config.model_revision,
            trust_remote_code=True,
            worker_extension_cls="open_instruct.vllm_utils_workerwrap.WorkerWrap",
            tensor_parallel_size=args.vllm_tensor_parallel_size,
            enforce_eager=args.vllm_enforce_eager,
            dtype="bfloat16",
            seed=args.seed + i,
            distributed_executor_backend=None,  # No distributed backend for local
            enable_prefix_caching=args.vllm_enable_prefix_caching,
            max_model_len=max_model_len,
            gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            num_gpus=1,
            noset_visible_devices=False,
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
            eval_results_queue=evaluation_inference_results_Q,
            actor_manager=actor_manager,
            tools=tool_objects,  # Use the tool objects we created
            max_tool_calls=max_tool_calls_dict,  # Use the converted dictionary
            inference_batch_size=args.inference_batch_size,  # Add the missing parameter
        )
        vllm_actors.append(llm_actor)
        logger.info(f"LLMRayActor {i} created successfully")
    logger.info("All LLMRayActor instances initialized successfully")
    
    # Setup training data iterator
    train_dataset_idxs = np.arange(len(train_dataset))
    iter_dataloader = grpo_fast.ShufflingIterator(train_dataset_idxs, args.num_unique_prompts_rollout, seed=args.seed)
    
    # Create pending queries maps
    pending_queries_map = grpo_fast.PendingQueriesMap()
    eval_pending_queries_map = grpo_fast.PendingQueriesMap()
    
    # Prepare eval batch if needed
    if eval_dataset is None:
        eval_batch = None
    else:
        eval_dataset_indices = list(range(min(32, len(eval_dataset))))  # Use 32 eval samples
        eval_batch = grpo_fast.next_batch(eval_dataset_indices, eval_dataset)
    
    # Create reward function
    reward_fn = grpo_fast.make_reward_fn(args)
    
    # Start threads with proper error handling
    stop_event = threading.Event()
    executor = futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="grpo")
    
    try:
        logger.info("======== ✅ data preparation thread starts =========")
        packing_future = executor.submit(
            grpo_fast.data_preparation_thread,
            reward_fn,
            inference_results_Q,
            packed_sequences_Q,
            pending_queries_map,
            args,
            tokenizer,
            args.num_training_steps,
            generation_configs["train"],
        )
        
        logger.info("======== ✅ generation thread starts =========")
        generation_future = executor.submit(
            local_generate_thread, vllm_actors, stop_event
        )
        
        # Send initial data to ensure we have a N-step offset
        for _ in range(args.async_steps):
            dataset_indices = next(iter_dataloader)
            batch = grpo_fast.next_batch(dataset_indices, train_dataset)
            grpo_fast.split_and_insert_batch(
                batch,
                1,  # All initial batches labeled as step 1
                args.vllm_num_engines,
                pending_queries_map,
                param_prompt_Q,
                generation_configs["train"],
            )
        
        # Token counting and main loop
        num_total_tokens = 0
        start_time = time.time()
        
        # Progress bar for token counting
        pbar = tqdm(total=args.num_training_steps, desc="Training steps")
        
        for training_step in range(1, args.num_training_steps + 1):
            # Check thread health by checking if they're still running
            if packing_future.done():
                # Thread died, check if it raised an exception
                try:
                    packing_future.result(timeout=0.1)
                except Exception as e:
                    logger.error(f"Data preparation thread died with error: {e}")
                    stop_event.set()
                    raise RuntimeError(f"Data preparation thread failed: {e}") from e
            
            if generation_future.done():
                # Thread died, check if it raised an exception
                try:
                    generation_future.result(timeout=0.1)
                except Exception as e:
                    logger.error(f"Generation thread died with error: {e}")
                    stop_event.set()
                    raise RuntimeError(f"Generation thread failed: {e}") from e
            
            # Prepare next batch of prompts (no weight sync needed in local mode)
            dataset_indices = next(iter_dataloader)
            batch = grpo_fast.next_batch(dataset_indices, train_dataset)
            grpo_fast.split_and_insert_batch(
                batch,
                training_step,  # Current training step, not future
                args.vllm_num_engines,
                pending_queries_map,
                param_prompt_Q,
                generation_configs["train"],
            )
            
            # Get packed data from packing thread
            collated_data, data_thread_metrics, num_total_tokens = grpo_fast.load_data_from_packing_thread(
                packed_sequences_Q, num_total_tokens, stop_event
            )
            
            if collated_data is None:
                continue
            
            # Update progress bar with token counts
            elapsed_time = time.time() - start_time
            tokens_per_sec = num_total_tokens / elapsed_time if elapsed_time > 0 else 0
            
            pbar.set_postfix({
                "Total tokens": num_total_tokens,
                "Tokens/sec": f"{tokens_per_sec:.1f}",
                "Batch score": f"{data_thread_metrics.get('scores', 0):.3f}",
            })
            pbar.update(1)
            
            # Log statistics periodically
            if training_step % 5 == 0:
                logger.info(
                    f"Step {training_step}: "
                    f"Total tokens: {num_total_tokens}, "
                    f"Tokens/sec: {tokens_per_sec:.1f}, "
                    f"Scores: {data_thread_metrics.get('scores', 0):.3f}"
                )
        
        pbar.close()
        
        # Final statistics
        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info("Final Statistics:")
        logger.info(f"Total training steps: {args.num_training_steps}")
        logger.info(f"Total tokens generated: {num_total_tokens}")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Average tokens/sec: {num_total_tokens / total_time:.1f}")
        logger.info("=" * 50)
    
    finally:
        # Clean up using the same cleanup function as grpo_fast.py
        logger.info("Cleaning up...")
        queues = [inference_results_Q, param_prompt_Q, evaluation_inference_results_Q]
        grpo_fast.cleanup_training_resources(stop_event, executor, queues, actor_manager)
        
        # Shutdown Ray
        ray.shutdown()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    # Parse arguments
    parser = ArgumentParserPlus((grpo_fast.Args, TokenizerConfig, ModelConfig))
    args, tokenizer_config, model_config = parser.parse_args_into_dataclasses()
    
    # Validate types
    assert isinstance(args, grpo_fast.Args)
    assert isinstance(tokenizer_config, TokenizerConfig)
    assert isinstance(model_config, ModelConfig)
    
    # Run main
    main(args, tokenizer_config, model_config)
