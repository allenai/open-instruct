import queue
import threading
import time
import unittest
import os
import logging
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import ray
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from open_instruct.grpo_fast import ShufflingIterator, vllm_generate_thread
from open_instruct.vllm_utils3 import create_vllm_engines
from ray.util import placement_group

# Set up logger
logger = logging.getLogger(__name__)


@dataclass
class MockCompletionOutput:
    """Mock completion output that mimics vLLM's output structure."""

    token_ids: List[int]
    finish_reason: str = "stop"  # "stop" or "length"
    # Tool-specific attributes
    mask: Optional[List[int]] = None
    num_calls: int = 0
    timeout: bool = False
    tool_error: str = ""
    tool_output: str = ""
    tool_runtime: float = 0.0
    tool_called: bool = False


@dataclass
class MockRequestOutput:
    """Mock request output that contains multiple completion outputs."""

    outputs: List[MockCompletionOutput]


class MockVLLMEngine:
    """Mock VLLM engine that simulates text generation."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.call_count = 0

    def generate(self, prompt_token_ids, sampling_params, use_tqdm=False):
        """Mock the remote generate method that returns a future-like object."""
        # Create a mock future that can be used with ray.get
        future = MagicMock()

        # Generate mock outputs for each prompt
        outputs = []
        for prompt_ids in prompt_token_ids:
            # Generate some fake token IDs (just append some tokens to the prompt)
            generated_tokens = list(range(100, 100 + sampling_params.max_tokens))
            all_tokens = prompt_ids + generated_tokens

            # Determine finish reason based on max_tokens
            finish_reason = "length" if len(generated_tokens) >= sampling_params.max_tokens else "stop"

            # Create completion output
            completion = MockCompletionOutput(
                token_ids=all_tokens,
                finish_reason=finish_reason,
                mask=[1] * len(all_tokens),  # All tokens are valid
                num_calls=0,
                timeout=False,
                tool_error="",
                tool_output="",
                tool_runtime=0.0,
                tool_called=False,
            )

            outputs.append(MockRequestOutput(outputs=[completion]))

        future.result = outputs
        self.call_count += 1
        return future


class MockVLLMEngineActor:
    """Mock Ray actor for VLLM engine."""

    def __init__(self, model_name: str):
        self.engine = MockVLLMEngine(model_name)
        # Create generate attribute with remote method
        self.generate = MagicMock()
        self.generate.remote = lambda prompt_token_ids, sampling_params, use_tqdm=False: self.engine.generate(
            prompt_token_ids, sampling_params, use_tqdm
        )


class TestGrpoFast(unittest.TestCase):
    def test_queue_based_generator_communication(self):
        """Test vllm_generate_thread with a real vLLM engine using a tiny model.

        This test requires CUDA availability for running real vLLM models.
        If CUDA is not available, this test will be skipped.
        """
        if not torch.cuda.is_available():
            self.skipTest("Skipping real vLLM test - CUDA not available")
        # Initialize ray if not already initialized
        if not ray.is_initialized():
            ray.init(num_cpus=2)

        try:
            # Create queues
            inference_results_Q = queue.Queue(maxsize=2)
            param_prompt_Q = queue.Queue(maxsize=2)
            evaluation_inference_results_Q = queue.Queue(maxsize=2)

            model_name = "allenai/OLMo-1B-hf"

            # Get tokenizer to create real token IDs
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create vLLM engines using the same function as grpo_fast.py
            bundles = [{"GPU": 1, "CPU": 10}]
            pg = placement_group.placement_group(bundles, strategy="STRICT_SPREAD")
            vllm_engines = create_vllm_engines(
                num_engines=1,
                tensor_parallel_size=1,
                enforce_eager=True,
                tokenizer_name_or_path=model_name,
                pretrain=model_name,
                revision=None,
                seed=42,
                enable_prefix_caching=False,
                max_model_len=512,  # Small context for testing
                vllm_gpu_memory_utilization=0.9,
                single_gpu_mode=True,
                pg=pg,
                tools={},
                max_tool_calls=0,
            )

            # Create test prompts
            test_texts = ["Hello world", "Testing vLLM"]
            test_prompts = [tokenizer.encode(text) for text in test_texts]

            # Start the vllm_generate_thread
            thread = threading.Thread(
                target=vllm_generate_thread,
                args=(
                    vllm_engines,  # vllm_engines list from create_vllm_engines
                    SamplingParams(temperature=0.0, max_tokens=10),  # generation_config - deterministic, short
                    SamplingParams(temperature=0.0, max_tokens=10),  # eval_generation_config
                    inference_results_Q,
                    param_prompt_Q,
                    1,  # num_training_steps - just one step for testing
                    None,  # eval_prompt_token_ids
                    evaluation_inference_results_Q,
                    2,  # eval_freq
                    1,  # resume_training_step
                    False,  # tool_use
                ),
            )
            thread.start()

            # Send prompts through the queue (expects tuple with (None, prompts))
            param_prompt_Q.put((None, test_prompts))

            # Get results
            try:
                result = inference_results_Q.get(timeout=30)  # Longer timeout for CPU generation

                # vllm_generate_thread returns a tuple: (response_ids, finish_reasons, masks, info)
                self.assertIsInstance(result, tuple)
                self.assertEqual(len(result), 4)

                response_ids, finish_reasons, masks, info = result

                # Verify the structure
                self.assertEqual(len(response_ids), len(test_prompts))
                self.assertEqual(len(finish_reasons), len(test_prompts))
                self.assertEqual(len(masks), len(test_prompts))

                # Each response should be a list of token IDs
                for resp in response_ids:
                    self.assertIsInstance(resp, list)
                    self.assertTrue(all(isinstance(token_id, int) for token_id in resp))
                    self.assertGreater(len(resp), 0)  # Should have generated something

                # Finish reasons should be strings
                for reason in finish_reasons:
                    self.assertIn(reason, ["stop", "length"])

                # Send None to stop the thread
                param_prompt_Q.put(None)

            finally:
                # Wait for thread to complete
                thread.join(timeout=5)

            # Clean up the actors
            for engine in vllm_engines:
                ray.kill(engine)

        finally:
            # Cleanup ray
            ray.shutdown()

    def test_multiple_queue_pipeline(self):
        """Test a pipeline with multiple queues like in grpo_fast."""
        # Create queues similar to grpo_fast
        queries_queue = queue.Queue(maxsize=2)
        request_queue = queue.Queue(maxsize=2)
        response_queue = queue.Queue(maxsize=2)
        packed_queue = queue.Queue(maxsize=2)

        # Test data
        queries = ["query1", "query2", "query3"]

        # Use a stop event to cleanly shut down threads
        stop_event = threading.Event()

        def data_prep_thread():
            """Simulates data_preparation_thread from grpo_fast."""
            while not stop_event.is_set():
                try:
                    query = queries_queue.get(timeout=0.1)
                    if query is None:
                        break

                    # Send to inference
                    request_queue.put({"query": query, "params": {}})

                    # Get inference result
                    result = response_queue.get(timeout=1)

                    # Pack and send to packed queue
                    packed = {"query": query, "result": result["response"]}
                    packed_queue.put(packed)

                except queue.Empty:
                    continue

        def inference_thread():
            """Simulates vllm_generate_thread from grpo_fast."""
            while not stop_event.is_set():
                try:
                    item = request_queue.get(timeout=0.1)
                    if isinstance(item, dict) and "query" in item:
                        # Simulate generation
                        response = f"Response for {item['query']}"
                        response_queue.put({"response": response})
                except queue.Empty:
                    continue

        # Start threads
        threads = [threading.Thread(target=data_prep_thread), threading.Thread(target=inference_thread)]

        for t in threads:
            t.start()

        # Send queries
        for query in queries:
            queries_queue.put(query)

        # Collect results
        results = []
        for _ in queries:
            try:
                result = packed_queue.get(timeout=2)
                results.append(result)
            except queue.Empty:
                break

        # Send sentinel and stop threads
        queries_queue.put(None)
        stop_event.set()

        for t in threads:
            t.join(timeout=2)

        # Verify results
        self.assertEqual(len(results), len(queries))
        for i, result in enumerate(results):
            self.assertEqual(result["query"], queries[i])
            self.assertEqual(result["result"], f"Response for {queries[i]}")

    @patch("ray.get")
    def test_queue_based_generator_communication_mocked(self, mock_ray_get):
        """Test vllm_generate_thread with mocked vLLM engines."""

        # Setup mock ray.get to return the result from our mock futures
        def mock_get_implementation(futures):
            if isinstance(futures, list):
                return [f.result for f in futures]
            return futures.result

        mock_ray_get.side_effect = mock_get_implementation

        # Create queues
        inference_results_Q = queue.Queue(maxsize=2)
        param_prompt_Q = queue.Queue(maxsize=2)
        evaluation_inference_results_Q = queue.Queue(maxsize=2)

        # Create mock vLLM engine actors
        model_name = "mock-model"
        vllm_engines = [MockVLLMEngineActor(model_name)]

        # Create test prompts (mock token IDs)
        test_prompts = [
            [1, 2, 3, 4, 5],  # "Hello world" equivalent
            [6, 7, 8, 9],  # "Testing vLLM" equivalent
        ]

        # Create mock sampling params
        generation_config = MagicMock()
        generation_config.temperature = 0.0
        generation_config.max_tokens = 10

        eval_generation_config = MagicMock()
        eval_generation_config.temperature = 0.0
        eval_generation_config.max_tokens = 10

        # Start the vllm_generate_thread
        thread = threading.Thread(
            target=vllm_generate_thread,
            args=(
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
            ),
        )
        thread.start()

        # Send prompts through the queue
        param_prompt_Q.put((None, test_prompts))

        # Get results
        try:
            result = inference_results_Q.get(timeout=5)

            # vllm_generate_thread returns: (response_ids, finish_reasons, masks, info)
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 4)

            response_ids, finish_reasons, masks, info = result

            # Verify the structure
            self.assertEqual(len(response_ids), len(test_prompts))
            self.assertEqual(len(finish_reasons), len(test_prompts))
            self.assertEqual(len(masks), len(test_prompts))

            # Each response should be a list of token IDs
            for i, resp in enumerate(response_ids):
                self.assertIsInstance(resp, list)
                self.assertTrue(all(isinstance(token_id, int) for token_id in resp))
                # Should contain original prompt + generated tokens
                self.assertGreater(len(resp), len(test_prompts[i]))
                # Check that prompt is preserved
                self.assertEqual(resp[: len(test_prompts[i])], test_prompts[i])

            # Finish reasons should be strings
            for reason in finish_reasons:
                self.assertIn(reason, ["stop", "length"])

            # Masks should be lists of integers
            for mask in masks:
                self.assertIsInstance(mask, list)
                self.assertTrue(all(isinstance(m, int) for m in mask))

            # Send None to stop the thread
            param_prompt_Q.put(None)

        finally:
            # Wait for thread to complete
            thread.join(timeout=5)

        # Verify ray.get was called
        mock_ray_get.assert_called()

    @patch("ray.get")
    def test_vllm_generate_with_tool_use_mocked(self, mock_ray_get):
        """Test vllm_generate_thread with tool_use enabled using mocks."""

        # Setup mock ray.get
        def mock_get_implementation(futures):
            if isinstance(futures, list):
                return [f.result for f in futures]
            return futures.result

        mock_ray_get.side_effect = mock_get_implementation

        # Create queues
        inference_results_Q = queue.Queue(maxsize=2)
        param_prompt_Q = queue.Queue(maxsize=2)
        evaluation_inference_results_Q = queue.Queue(maxsize=2)

        # Create mock vLLM engine with tool support
        class ToolMockVLLMEngine(MockVLLMEngine):
            def generate(self, prompt_token_ids, sampling_params, use_tqdm=False):
                future = MagicMock()
                outputs = []

                for prompt_ids in prompt_token_ids:
                    generated_tokens = list(range(200, 210))
                    all_tokens = prompt_ids + generated_tokens

                    # Create completion with tool attributes
                    completion = MockCompletionOutput(
                        token_ids=all_tokens,
                        finish_reason="stop",
                        mask=[1] * len(all_tokens),
                        num_calls=1,
                        timeout=False,
                        tool_error="",
                        tool_output="Tool execution result",
                        tool_runtime=0.5,
                        tool_called=True,
                    )

                    outputs.append(MockRequestOutput(outputs=[completion]))

                future.result = outputs
                return future

        # Create mock engine actor with tool support
        class ToolMockVLLMEngineActor(MockVLLMEngineActor):
            def __init__(self, model_name: str):
                self.engine = ToolMockVLLMEngine(model_name)
                # Create generate attribute with remote method
                self.generate = MagicMock()
                self.generate.remote = lambda prompt_token_ids, sampling_params, use_tqdm=False: self.engine.generate(
                    prompt_token_ids, sampling_params, use_tqdm
                )

        vllm_engines = [ToolMockVLLMEngineActor("mock-model")]

        # Test prompts
        test_prompts = [[1, 2, 3], [4, 5, 6]]

        # Mock sampling params
        generation_config = MagicMock()
        generation_config.temperature = 0.0
        generation_config.max_tokens = 10

        eval_generation_config = MagicMock()
        eval_generation_config.temperature = 0.0
        eval_generation_config.max_tokens = 10

        # Start thread with tool_use=True
        thread = threading.Thread(
            target=vllm_generate_thread,
            args=(
                vllm_engines,
                generation_config,
                eval_generation_config,
                inference_results_Q,
                param_prompt_Q,
                1,
                None,
                evaluation_inference_results_Q,
                2,
                1,
                True,  # tool_use enabled
            ),
        )
        thread.start()

        # Send prompts
        param_prompt_Q.put((None, test_prompts))

        # Get results
        try:
            result = inference_results_Q.get(timeout=5)
            response_ids, finish_reasons, masks, info = result

            # info is a tuple: (num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds)
            self.assertIsInstance(info, tuple)
            self.assertEqual(len(info), 6)

            num_calls, timeouts, tool_errors, tool_outputs, tool_runtimes, tool_calleds = info

            # Verify tool attributes
            self.assertEqual(num_calls, [1, 1])
            self.assertEqual(timeouts, [False, False])
            self.assertEqual(tool_calleds, [True, True])
            self.assertEqual(tool_outputs, ["Tool execution result", "Tool execution result"])
            self.assertEqual(tool_errors, ["", ""])
            self.assertEqual(tool_runtimes, [0.5, 0.5])

            # Send None to stop
            param_prompt_Q.put(None)

        finally:
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
