import unittest
from unittest import mock

import ray
import vllm
from parameterized import parameterized
from ray.util import queue as ray_queue

from open_instruct import vllm_utils3


class MockVLLMOutput:
    """Mock vLLM output object."""

    def __init__(self, token_ids, finish_reason="stop"):
        self.token_ids = token_ids
        self.finish_reason = finish_reason


class MockVLLMResponse:
    """Mock vLLM response object."""

    def __init__(self, outputs):
        self.outputs = outputs


class MockLLM:
    """Mock LLM class that mimics vLLM interface."""

    def __init__(self, *args, **kwargs):
        # Store initialization parameters for verification
        self.model = kwargs.get("model")
        self.tokenizer = kwargs.get("tokenizer")
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        # Store llm_engine attribute for compatibility
        self.llm_engine = mock.Mock()
        self.llm_engine.reset_prefix_cache = mock.Mock()

    def generate(self, prompt_token_ids, sampling_params, **kwargs):
        """Mock generate method that returns code-like tokens."""
        responses = []
        for prompt in prompt_token_ids:
            # Generate mock code tokens based on prompt
            if 101 in prompt:  # Python code
                tokens = [1000, 1001, 1002, 1003, 1004]  # def function():
            elif 201 in prompt:  # JavaScript code
                tokens = [2000, 2001, 2002, 2003, 2004]  # function() {}
            elif 301 in prompt:  # SQL code
                tokens = [3000, 3001, 3002, 3003, 3004]  # SELECT * FROM
            else:
                tokens = [1, 2, 3, 4, 5]  # Default tokens

            output = MockVLLMOutput(tokens, "stop")
            responses.append(MockVLLMResponse([output]))
        return responses

    def collective_rpc(self, method_name, args):
        """Mock collective_rpc method."""
        return None

    def sleep(self, level=1):
        """Mock sleep method."""
        pass

    def wake_up(self):
        """Mock wake_up method."""
        pass


@ray.remote
class TestableRayActorWithMockVLLM:
    """Ray actor that uses real LLMRayActor logic but with mocked vLLM.

    This allows us to test the actual LLMRayActor code without requiring a GPU or real model.
    """

    def __init__(self, *args, **kwargs):
        # Import and mock vLLM locally in the actor process
        import vllm as vllm_module

        # Save original LLM class
        original_llm = vllm_module.LLM

        # Replace with mock
        vllm_module.LLM = MockLLM

        # Now we need to get the underlying class from the Ray actor
        # LLMRayActor is decorated with @ray.remote, so we need to access the wrapped class
        from open_instruct import vllm_utils3

        # Access the underlying class from the Ray ActorClass
        LLMRayActorClass = vllm_utils3.LLMRayActor.__ray_actor_class__

        # Create an instance directly (not as a Ray actor since we're already in one)
        self._actor = LLMRayActorClass(*args, **kwargs)

        # Restore original (though it won't affect the already created instance)
        vllm_module.LLM = original_llm

    def generate(self, *args, **kwargs):
        """Delegate to the real LLMRayActor generate method."""
        return self._actor.generate(*args, **kwargs)

    def process_from_queue(self, *args, **kwargs):
        """Delegate to the real LLMRayActor process_from_queue method."""
        return self._actor.process_from_queue(*args, **kwargs)

    def ready(self):
        """Delegate to the real LLMRayActor ready method."""
        return self._actor.ready()

    def reset_prefix_cache(self):
        """Delegate to the real LLMRayActor reset_prefix_cache method."""
        return self._actor.reset_prefix_cache()

    def init_process_group(self, *args, **kwargs):
        """Delegate to the real LLMRayActor init_process_group method."""
        return self._actor.init_process_group(*args, **kwargs)

    def update_weight(self, *args, **kwargs):
        """Delegate to the real LLMRayActor update_weight method."""
        return self._actor.update_weight(*args, **kwargs)

    def sleep(self, *args, **kwargs):
        """Delegate to the real LLMRayActor sleep method."""
        return self._actor.sleep(*args, **kwargs)

    def wake_up(self):
        """Delegate to the real LLMRayActor wake_up method."""
        return self._actor.wake_up()


class TestLLMRayActorTests(unittest.TestCase):
    """Test LLMRayActor with mocked vLLM."""

    @classmethod
    def setUpClass(cls):
        """Initialize Ray once for all tests."""
        if not ray.is_initialized():
            ray.init(include_dashboard=False)

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray after all tests."""
        if ray.is_initialized():
            ray.shutdown()

    @parameterized.expand(
        [
            ("python", [101, 102, 103], [1000, 1001, 1002, 1003, 1004]),
            ("javascript", [201, 202, 203], [2000, 2001, 2002, 2003, 2004]),
            ("sql", [301, 302, 303], [3000, 3001, 3002, 3003, 3004]),
            ("default", [401, 402, 403], [1, 2, 3, 4, 5]),
        ]
    )
    def test_llm_ray_actor_code_generation(self, name, prompt_tokens, expected_tokens):
        """Test real LLMRayActor can generate different types of code with mocked vLLM."""
        # Create actor that uses real LLMRayActor code with mocked vLLM
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,  # Use 0 GPUs for CPU testing
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Test generate method with code generation prompt
        code_prompt = [prompt_tokens]
        sampling_params = vllm.SamplingParams(temperature=0.7, max_tokens=100)

        # Call generate - this uses the real LLMRayActor.generate method
        result = ray.get(actor.generate.remote(prompt_token_ids=code_prompt, sampling_params=sampling_params))

        # Verify the result
        self.assertEqual(len(result), 1)
        self.assertEqual(len(result[0].outputs), 1)
        self.assertEqual(result[0].outputs[0].token_ids, expected_tokens)
        self.assertEqual(result[0].outputs[0].finish_reason, "stop")

    def test_llm_ray_actor_batch_generation(self):
        """Test real LLMRayActor can handle batch code generation."""
        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Test batch generation with multiple code prompts
        code_prompts = [
            [101, 102, 103],  # Prompt for Python function
            [201, 202, 203],  # Prompt for JavaScript function
            [301, 302, 303],  # Prompt for SQL query
        ]
        sampling_params = vllm.SamplingParams(temperature=0.5, max_tokens=50)

        # Call generate using real LLMRayActor
        result = ray.get(actor.generate.remote(prompt_token_ids=code_prompts, sampling_params=sampling_params))

        # Verify the results
        self.assertEqual(len(result), 3)
        # Python code tokens
        self.assertEqual(result[0].outputs[0].token_ids, [1000, 1001, 1002, 1003, 1004])
        # JavaScript code tokens
        self.assertEqual(result[1].outputs[0].token_ids, [2000, 2001, 2002, 2003, 2004])
        # SQL code tokens
        self.assertEqual(result[2].outputs[0].token_ids, [3000, 3001, 3002, 3003, 3004])

    @parameterized.expand(
        [
            ("training", False, [101, 102, 103], [1000, 1001, 1002, 1003, 1004], 1),
            ("evaluation", True, [201, 202, 203], [2000, 2001, 2002, 2003, 2004], 2),
        ]
    )
    def test_llm_ray_actor_with_queues(self, name, is_eval, prompt_tokens, expected_tokens, training_step):
        """Test real LLMRayActor processing requests from queue."""
        # Create Ray queues
        prompt_queue = ray_queue.Queue()
        results_queue = ray_queue.Queue()
        eval_results_queue = ray_queue.Queue()

        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=prompt_queue,
            results_queue=results_queue,
            eval_results_queue=eval_results_queue,
        )

        # Create a prompt request
        prompt_request = vllm_utils3.PromptRequest(
            prompts=[prompt_tokens],
            generation_config=vllm.SamplingParams(temperature=0.8, max_tokens=75),
            dataset_index=[training_step - 1],
            training_step=training_step,
            is_eval=is_eval,
        )

        # Put request in queue
        prompt_queue.put(prompt_request)

        # Process from queue using real LLMRayActor.process_from_queue
        processed = ray.get(actor.process_from_queue.remote(timeout=1.0))
        self.assertEqual(processed, 1)  # Successfully processed one request

        # Check appropriate results queue
        target_queue = eval_results_queue if is_eval else results_queue
        result = target_queue.get(timeout=1.0)
        self.assertIsInstance(result, vllm_utils3.GenerationResult)
        self.assertEqual(result.responses, [expected_tokens])
        self.assertEqual(result.finish_reasons, ["stop"])
        self.assertEqual(result.dataset_index, [training_step - 1])
        self.assertEqual(result.training_step, training_step)

        # Verify masks are generated correctly
        self.assertEqual(result.masks, [[1] * len(expected_tokens)])

        # Verify request_info is populated correctly
        self.assertIsInstance(result.request_info, vllm_utils3.RequestInfo)
        self.assertEqual(result.request_info.num_calls, [0])
        self.assertEqual(result.request_info.tool_calleds, [False])

        # Shutdown queues
        prompt_queue.shutdown()
        results_queue.shutdown()
        eval_results_queue.shutdown()

    def test_llm_ray_actor_ready_check(self):
        """Test real LLMRayActor ready check."""
        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Check if actor is ready using real ready method
        is_ready = ray.get(actor.ready.remote())
        self.assertTrue(is_ready)

    @parameterized.expand([("small_batch", 2, 0.5, 50), ("large_batch", 5, 0.8, 100), ("single_item", 1, 0.3, 25)])
    def test_llm_ray_actor_batch_sizes(self, name, batch_size, temperature, max_tokens):
        """Test real LLMRayActor with different batch sizes and parameters."""
        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Create batch of prompts
        code_prompts = [[101, 102, 103] for _ in range(batch_size)]
        sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=max_tokens)

        # Call generate using real LLMRayActor
        result = ray.get(actor.generate.remote(prompt_token_ids=code_prompts, sampling_params=sampling_params))

        # Verify the results
        self.assertEqual(len(result), batch_size)
        for i in range(batch_size):
            self.assertEqual(result[i].outputs[0].token_ids, [1000, 1001, 1002, 1003, 1004])
            self.assertEqual(result[i].outputs[0].finish_reason, "stop")

    def test_llm_ray_actor_reset_prefix_cache(self):
        """Test real LLMRayActor reset_prefix_cache method."""
        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Test reset_prefix_cache - should not raise an error
        ray.get(actor.reset_prefix_cache.remote())

        # Verify the method exists and can be called
        self.assertTrue(True)  # If we get here, the method exists and works

    def test_llm_ray_actor_collective_methods(self):
        """Test real LLMRayActor collective RPC methods."""
        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Test init_process_group
        result = ray.get(
            actor.init_process_group.remote(
                master_address="127.0.0.1",
                master_port=12345,
                rank_offset=0,
                world_size=1,
                group_name="test_group",
                backend="gloo",
                use_ray=False,
            )
        )
        self.assertIsNone(result)  # MockLLM returns None for collective_rpc

        # Test update_weight
        result = ray.get(
            actor.update_weight.remote(name="test_weight", dtype="float32", shape=[10, 10], empty_cache=False)
        )
        self.assertIsNone(result)  # MockLLM returns None for collective_rpc

    def test_llm_ray_actor_sleep_wake(self):
        """Test real LLMRayActor sleep and wake_up methods."""
        # Create actor with real LLMRayActor code
        actor = TestableRayActorWithMockVLLM.remote(
            model="mock-model",
            tokenizer="mock-tokenizer",
            trust_remote_code=True,
            tensor_parallel_size=1,
            enforce_eager=True,
            dtype="bfloat16",
            seed=42,
            distributed_executor_backend="uni",
            enable_prefix_caching=False,
            max_model_len=1024,
            gpu_memory_utilization=0.9,
            num_gpus=0,
            noset_visible_devices=False,
            tool_use=False,
            prompt_queue=None,
            results_queue=None,
            eval_results_queue=None,
        )

        # Test sleep
        ray.get(actor.sleep.remote(level=1))

        # Test wake_up
        ray.get(actor.wake_up.remote())

        # If we get here, the methods exist and can be called
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
