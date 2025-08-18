import gc
import os
import unittest

import ray
import torch
from ray.util import queue as ray_queue
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct import utils
from open_instruct.queue_types import GenerationResult, PromptRequest
from open_instruct.vllm_utils3 import create_vllm_engines


class TestGrpoFastGPUBase(unittest.TestCase):
    """Base class with common test utilities for GPU tests."""

    def _get_resource_tracker_state(self):
        """Get current resource tracker state for debugging."""
        tracked_resources = {}
        try:
            # Try to access resource tracker directly
            from multiprocessing.resource_tracker import _resource_tracker

            if hasattr(_resource_tracker, "_cache"):
                for name, rtype in list(_resource_tracker._cache.items()):
                    if rtype not in tracked_resources:
                        tracked_resources[rtype] = []
                    tracked_resources[rtype].append(name)
        except Exception:
            # Alternative approach: check via resource_tracker module
            try:
                import multiprocessing.resource_tracker as rt

                if hasattr(rt, "getfd"):
                    # This is a hack to get the cache info

                    # Try to find the cache in the module
                    for attr_name in dir(rt):
                        attr = getattr(rt, attr_name)
                        if isinstance(attr, dict) and any("semaphore" in str(v) for v in attr.values()):
                            for k, v in attr.items():
                                if v not in tracked_resources:
                                    tracked_resources[v] = []
                                tracked_resources[v].append(k)
            except Exception:
                pass
        return tracked_resources

    def setUp(self):
        """Initialize Ray and check for pre-existing leaks."""
        # Check if CUDA is available
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, skipping test")

        # Save original environment variable value
        self._original_nccl_cumem = os.environ.get("NCCL_CUMEM_ENABLE")

        # Record initial resource tracker state
        self._initial_resources = self._get_resource_tracker_state()

        # Track Ray queues for cleanup
        self._ray_queues = []

        # Check for leaks after Ray init
        leak_report = utils.check_runtime_leaks()
        # After Ray init, we expect exactly one Ray head worker
        if len(leak_report.ray_workers) == 1:
            # Check if it's the head worker (worker ID all zeros or all f's)
            worker = leak_report.ray_workers[0]
            worker_id = worker.get("worker_id", "")
            if worker_id in [
                "01000000ffffffffffffffffffffffffffffffffffffffffffffffff",
                "00000000ffffffffffffffffffffffffffffffffffffffffffffffff",
            ]:
                # This is the expected Ray head worker, clear it
                leak_report.ray_workers = []

        if not leak_report.is_clean:
            self.fail(f"Leaks detected before test {self._testMethodName}:\n{leak_report.pretty()}")

        # Initialize Ray for this test
        ray.init(include_dashboard=False)

    def _cleanup_ray_queues(self):
        """Clean up all Ray queues created during the test."""
        for queue in self._ray_queues:
            try:
                queue.shutdown()
            except Exception as e:
                print(f"Warning: Failed to shutdown Ray queue: {e}")
        self._ray_queues.clear()

    def tearDown(self):
        """Check for leaks and shutdown Ray."""
        # Clean up Ray queues BEFORE shutting down Ray
        self._cleanup_ray_queues()

        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

        # Force garbage collection to clean up any lingering objects
        gc.collect()

        # Get final resource tracker state
        final_resources = self._get_resource_tracker_state()

        # Check for new resources that weren't there initially
        new_resources = {}
        for rtype, names in final_resources.items():
            initial_names = set(self._initial_resources.get(rtype, []))
            new_names = [n for n in names if n not in initial_names]
            if new_names:
                new_resources[rtype] = new_names

        # Check for leaks before shutdown
        leak_report = utils.check_runtime_leaks()
        # We still expect the Ray head worker
        if len(leak_report.ray_workers) == 1:
            worker = leak_report.ray_workers[0]
            worker_id = worker.get("worker_id", "")
            if worker_id in [
                "01000000ffffffffffffffffffffffffffffffffffffffffffffffff",
                "00000000ffffffffffffffffffffffffffffffffffffffffffffffff",
            ]:
                # This is the expected Ray head worker, clear it
                leak_report.ray_workers = []

        if not leak_report.is_clean:
            self.fail(f"Leaks detected after test {self._testMethodName}:\n{leak_report.pretty()}")

        # Check for semaphore leaks
        if new_resources:
            # Report all new resources, especially semaphores
            leak_msg = f"Resource leaks detected after test {self._testMethodName}:\n"
            for rtype, names in new_resources.items():
                leak_msg += f"  {rtype}: {names}\n"

            # Fail if there are semaphore leaks
            if "semaphore" in new_resources:
                self.fail(leak_msg)

        # Restore original environment variable value
        if self._original_nccl_cumem is None:
            os.environ.pop("NCCL_CUMEM_ENABLE", None)
        else:
            os.environ["NCCL_CUMEM_ENABLE"] = self._original_nccl_cumem


class TestGrpoFastVLLMGPU(TestGrpoFastGPUBase):
    def test_vllm_queue_system_single_prompt(self):
        """Test the new queue-based vLLM system with a single prompt 'What is the capital of France?'"""
        # Set up tokenizer
        tokenizer_name = "EleutherAI/pythia-14m"  # Using a small model for testing
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Tokenize the test prompt
        test_prompt = "What is the capital of France?"
        prompt_token_ids = tokenizer.encode(test_prompt, return_tensors="pt").tolist()[0]

        # Create Ray queues
        param_prompt_Q = ray_queue.Queue(maxsize=1)
        inference_results_Q = ray_queue.Queue(maxsize=1)

        # Track queues for cleanup
        self._ray_queues.extend([param_prompt_Q, inference_results_Q])

        # Create vLLM engines with queues
        _ = create_vllm_engines(
            num_engines=1,
            tensor_parallel_size=1,
            enforce_eager=True,
            tokenizer_name_or_path=tokenizer_name,
            pretrain=tokenizer_name,
            revision="main",
            seed=42,
            enable_prefix_caching=False,
            max_model_len=512,
            vllm_gpu_memory_utilization=0.5,  # Use less GPU memory for testing
            prompt_queue=param_prompt_Q,
            results_queue=inference_results_Q,
        )

        # Set up generation config
        generation_config = SamplingParams(
            temperature=0.0,  # Deterministic generation
            top_p=1.0,
            max_tokens=5,
            n=1,
        )

        # Create a PromptRequest
        request = PromptRequest(
            prompt_token_ids=prompt_token_ids, generation_config=generation_config, dataset_index=[0], training_step=0
        )

        # Send the request
        param_prompt_Q.put(request)

        # Get the result
        result = inference_results_Q.get(timeout=30)

        # Verify we got a GenerationResult
        self.assertIsInstance(result, GenerationResult)
        self.assertIsNotNone(result.responses)
        self.assertEqual(len(result.responses), 1)
        self.assertEqual(result.dataset_index, [0])

        # Get the response IDs (skip the prompt)
        response_ids = result.responses[0]

        # Decode the response
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)

        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), 0)

        # Send stop signal
        param_prompt_Q.put(None)
