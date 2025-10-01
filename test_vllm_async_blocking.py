#!/usr/bin/env python
"""Test script to reproduce vLLM async blocking issue with high concurrency."""

import asyncio
import time
import argparse
import logging
from typing import List
import vllm
from vllm import AsyncLLMEngine, SamplingParams, TokensPrompt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def process_single_request(
    engine: AsyncLLMEngine,
    request_id: str,
    prompt_tokens: List[int],
    sampling_params: SamplingParams,
    request_index: int,
) -> dict:
    """Process a single request and track timing."""
    result = {
        'request_id': request_id,
        'index': request_index,
        'start_time': time.perf_counter(),
        'add_request_time': None,
        'first_output_time': None,
        'finish_time': None,
        'iteration_count': 0,
        'completed': False,
        'error': None
    }

    try:
        logger.info(f"[{request_id}] Starting request")

        # Create prompt
        tokens_prompt = TokensPrompt(prompt_token_ids=prompt_tokens)

        # Add request and get generator
        generator = await engine.add_request(
            request_id=request_id,
            prompt=tokens_prompt,
            params=sampling_params
        )

        result['add_request_time'] = time.perf_counter()
        logger.info(f"[{request_id}] Got generator, starting iteration")

        # Iterate over outputs
        async for output in generator:
            result['iteration_count'] += 1

            if result['iteration_count'] == 1:
                result['first_output_time'] = time.perf_counter()
                logger.info(f"[{request_id}] Got first output after {result['first_output_time'] - result['start_time']:.2f}s")

            if result['iteration_count'] % 100 == 0:
                logger.info(f"[{request_id}] Iteration {result['iteration_count']}")

            if output.finished:
                result['finish_time'] = time.perf_counter()
                result['completed'] = True
                logger.info(f"[{request_id}] Completed after {result['iteration_count']} iterations in {result['finish_time'] - result['start_time']:.2f}s")
                break

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"[{request_id}] Error: {e}")

    return result


async def event_loop_monitor():
    """Monitor if event loop is responsive."""
    iteration = 0
    while True:
        iteration += 1
        start = time.perf_counter()
        await asyncio.sleep(1.0)
        elapsed = time.perf_counter() - start

        if elapsed > 1.5:  # Should be ~1.0s
            logger.warning(f"⚠️ Event loop blocked! Sleep took {elapsed:.2f}s instead of 1.0s")
        else:
            logger.debug(f"✅ Event loop responsive (iteration {iteration})")


async def run_test(
    model_name: str,
    num_prompts: int,
    num_samples_per_prompt: int,
    prompt_length: int,
    max_tokens: int,
    tensor_parallel_size: int = 1,
):
    """Run the async blocking test."""
    total_requests = num_prompts * num_samples_per_prompt
    logger.info(f"Starting test with {total_requests} total requests ({num_prompts} prompts × {num_samples_per_prompt} samples)")

    # Initialize engine
    logger.info(f"Initializing AsyncLLMEngine for {model_name}...")
    engine = AsyncLLMEngine.from_engine_args(
        vllm.AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=2048,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=True,  # Disable CUDA graphs
            disable_log_requests=True,
            disable_log_stats=True,
        ),
        start_engine_loop=True,
    )

    # Create dummy prompts (just repeated tokens)
    dummy_token = 1234  # Some valid token ID
    prompt_tokens = [dummy_token] * prompt_length

    # Start event loop monitor
    monitor_task = asyncio.create_task(event_loop_monitor())

    # Create all tasks at once (mimicking the issue)
    tasks = []
    logger.info(f"Creating {total_requests} async tasks...")

    for prompt_idx in range(num_prompts):
        sampling_params = SamplingParams(
            n=1,  # We handle n>1 by creating multiple tasks
            max_tokens=max_tokens,
            temperature=1.0,
            seed=42 + prompt_idx,
        )

        for sample_idx in range(num_samples_per_prompt):
            request_id = f"test_{prompt_idx}_{sample_idx}"

            # Create task immediately (like the original code)
            task = asyncio.create_task(
                process_single_request(
                    engine=engine,
                    request_id=request_id,
                    prompt_tokens=prompt_tokens,
                    sampling_params=sampling_params,
                    request_index=prompt_idx * num_samples_per_prompt + sample_idx,
                ),
                name=request_id
            )
            tasks.append(task)

    logger.info(f"All {len(tasks)} tasks created, waiting for completion...")

    # Wait for all tasks with timeout
    start_time = time.perf_counter()
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=120.0  # 2 minute timeout
        )
    except asyncio.TimeoutError:
        logger.error("❌ Timeout! Tasks did not complete within 120 seconds")
        results = []
        for task in tasks:
            if not task.done():
                task.cancel()

    elapsed = time.perf_counter() - start_time

    # Cancel monitor
    monitor_task.cancel()

    # Analyze results
    completed_count = sum(1 for r in results if isinstance(r, dict) and r.get('completed'))
    error_count = sum(1 for r in results if isinstance(r, dict) and r.get('error'))
    timeout_count = sum(1 for task in tasks if not task.done())

    logger.info("=" * 60)
    logger.info(f"Test completed in {elapsed:.2f}s")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Completed: {completed_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Timed out: {timeout_count}")

    # Show which indices completed
    if completed_count > 0:
        completed_indices = [r['index'] for r in results if isinstance(r, dict) and r.get('completed')]
        logger.info(f"Completed indices: {sorted(completed_indices)[:20]}...")  # Show first 20

    # Show timing stats for completed requests
    if completed_count > 0:
        completed_results = [r for r in results if isinstance(r, dict) and r.get('completed')]
        avg_time = sum(r['finish_time'] - r['start_time'] for r in completed_results) / len(completed_results)
        logger.info(f"Average completion time: {avg_time:.2f}s")

    # Cleanup
    engine.shutdown_background_loop()
    del engine

    return {
        'total': total_requests,
        'completed': completed_count,
        'errors': error_count,
        'timeout': timeout_count,
        'elapsed': elapsed
    }


async def main():
    parser = argparse.ArgumentParser(description='Test vLLM async blocking with high concurrency')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B', help='Model to use')
    parser.add_argument('--num-prompts', type=int, default=4, help='Number of unique prompts')
    parser.add_argument('--num-samples', type=int, default=16, help='Number of samples per prompt')
    parser.add_argument('--prompt-length', type=int, default=512, help='Length of prompt in tokens')
    parser.add_argument('--max-tokens', type=int, default=100, help='Max tokens to generate')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size')
    args = parser.parse_args()

    # Test with different concurrency levels
    test_configs = [
        (2, 16),  # 32 total requests (should work)
        (4, 16),  # 64 total requests (should hang?)
        (8, 16),  # 128 total requests (should definitely hang?)
    ]

    for num_prompts, num_samples in test_configs:
        logger.info("=" * 80)
        logger.info(f"Testing with {num_prompts} prompts × {num_samples} samples = {num_prompts * num_samples} total")
        logger.info("=" * 80)

        result = await run_test(
            model_name=args.model,
            num_prompts=num_prompts,
            num_samples_per_prompt=num_samples,
            prompt_length=args.prompt_length,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )

        # Stop if we hit a hang
        if result['timeout'] > 0:
            logger.warning(f"⚠️ Test hung with {num_prompts * num_samples} concurrent requests!")
            break
        else:
            logger.info(f"✅ Test passed with {num_prompts * num_samples} concurrent requests")

        # Small delay between tests
        await asyncio.sleep(5)


if __name__ == "__main__":
    asyncio.run(main())