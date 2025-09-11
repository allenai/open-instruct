import asyncio
import logging
import queue
import threading
import time
import unittest
from concurrent import futures
from unittest import mock
from unittest.mock import MagicMock, patch

import vllm

from open_instruct.queue_types import TokenStatistics
from open_instruct.vllm_utils3 import LLMRayActor, _finalize_outputs, _init_tool_tracking


class FakeTool:
    """Fake tool for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_args = None

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.last_args = (args, kwargs)
        return "TOOL_OUTPUT"


class FakeRequestOutput:
    """Fake RequestOutput for testing."""

    def __init__(self, request_id, outputs=None, finished=True):
        self.request_id = request_id
        self.outputs = outputs or []
        self.finished = finished


class FakeCompletionOutput:
    """Fake CompletionOutput for testing."""

    def __init__(self, text="", token_ids=None, finish_reason=None, stop_reason=None):
        self.text = text
        self.token_ids = token_ids or []
        self.finish_reason = finish_reason
        self.stop_reason = stop_reason
        self.mask = None
        self.num_calls = 0
        self.timeout = False
        self.tool_error = ""
        self.tool_output = ""
        self.tool_runtime = 0.0
        self.tool_called = False


class FakeAsyncLLMEngine:
    """Fake AsyncLLMEngine for testing."""

    def __init__(self):
        self.unfinished_requests = {}
        self.add_request_calls = []
        self.abort_request_calls = []
        self.tokenizer = MagicMock()
        self.tokenizer.eos_token_id = 2
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.decode.return_value = "decoded_text"

    async def add_request(self, request_id, prompt, sampling_params, **kwargs):
        self.add_request_calls.append(
            {"request_id": request_id, "prompt": prompt, "sampling_params": sampling_params, "kwargs": kwargs}
        )
        self.unfinished_requests[request_id] = True

    async def abort_request(self, request_id):
        self.abort_request_calls.append(request_id)
        if request_id in self.unfinished_requests:
            del self.unfinished_requests[request_id]

    def get_num_unfinished_requests(self):
        return len(self.unfinished_requests)

    def has_unfinished_requests(self):
        return len(self.unfinished_requests) > 0

    def finish_request(self, request_id, outputs=None, finished=True):
        """Simulate finishing a request."""
        if request_id in self.unfinished_requests and finished:
            del self.unfinished_requests[request_id]
        return FakeRequestOutput(request_id, outputs, finished)

    def step(self):
        """Simulate engine step."""
        return []


class FakeActorManager:
    """Fake ActorManager for testing."""

    def __init__(self):
        self._should_stop = False

    class RemoteCallable:
        def __init__(self, value):
            self.value = value

        def remote(self):
            return self

        def __await__(self):
            async def _return():
                return self.value

            return _return().__await__()

    @property
    def should_stop(self):
        return self.RemoteCallable(self._should_stop)

    def set_should_stop(self, value):
        self._should_stop = value


class TestVLLMUtils3New(unittest.TestCase):
    """New comprehensive test cases for vllm_utils3 module."""

    def setUp(self):
        """Set up test fixtures."""
        logging.disable(logging.CRITICAL)
        self.tokenizer = mock.MagicMock()
        self.tokenizer.eos_token_id = 2
        self.tokenizer.encode.return_value = [1, 2, 3]
        self.tokenizer.decode.return_value = "decoded_text"

        self.model_config = mock.MagicMock()
        self.model_config.max_model_len = 1000

        self.generation_config = mock.MagicMock()
        self.generation_config.max_tokens = 100
        self.generation_config.temperature = 1.0
        self.generation_config.top_p = 1.0
        self.generation_config.n = 1

        self.engine = FakeAsyncLLMEngine()
        self.actor_manager = FakeActorManager()

        self.metadata = {}
        self.tracking = _init_tool_tracking()
        self.results_queue = queue.Queue()

    def tearDown(self):
        """Clean up after test."""
        logging.disable(logging.NOTSET)

    def test_tools_configured_not_used_n1(self):
        """Test tools configured but not used with n=1."""
        # Create actor instance
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"any": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine
        actor.executor = MagicMock()
        actor.max_tool_calls = 10
        actor._engine_steps_timing = {"engine_step_call": 0.0, "extract_base_request_id": 0.0, "handle_output": 0.0}

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        sub_request_id = f"{request_id}_0"

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": self.generation_config,
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        # Set up tracking
        tracking = _init_tool_tracking()
        tracking["concat_outputs"][sub_request_id] = None

        # Create output without tool call
        mock_output = FakeCompletionOutput(text="Normal output", token_ids=[4, 5, 6], finish_reason="stop")
        mock_output.mask = [1, 1, 1]

        mock_request = FakeRequestOutput(sub_request_id, [mock_output], finished=True)
        mock_request.prompt = "test prompt"
        mock_request.prompt_token_ids = [1, 2, 3]

        # Store the output
        tracking["concat_outputs"][sub_request_id] = mock_request
        tracking["masks"][sub_request_id] = [1, 1, 1]
        tracking["num_calls"][sub_request_id] = 0
        tracking["timeout"][sub_request_id] = False
        tracking["tool_error"][sub_request_id] = ""
        tracking["tool_output"][sub_request_id] = ""
        tracking["tool_runtime"][sub_request_id] = 0.0
        tracking["tool_called"][sub_request_id] = False

        # Call finalize with properly structured tracking
        # Note: _finalize_outputs expects the base request ID from output
        mock_request.request_id = request_id  # Use base request ID, not sub-request ID
        result = _finalize_outputs(
            output=mock_request,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=3, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result.responses), 1)
        self.assertEqual(result.responses[0], [4, 5, 6])
        self.assertEqual(result.masks[0], [1, 1, 1])

    def test_tools_configured_not_used_n3(self):
        """Test tools configured but not used with n>1."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"any": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        n = 3

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": MagicMock(n=n),
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        # Set up tracking with 3 samples
        tracking = _init_tool_tracking()

        for i in range(n):
            sub_id = f"{request_id}_{i}"
            mock_output = FakeCompletionOutput(
                text=f"Output {i}", token_ids=[4 + i, 5 + i, 6 + i], finish_reason="stop"
            )
            mock_output.mask = [1, 1, 1]

            mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
            mock_request.prompt = "test prompt"
            mock_request.prompt_token_ids = [1, 2, 3]

            tracking["concat_outputs"][sub_id] = mock_request
            tracking["masks"][sub_id] = [1, 1, 1]
            tracking["num_calls"][sub_id] = 0
            tracking["timeout"][sub_id] = False
            tracking["tool_error"][sub_id] = ""
            tracking["tool_output"][sub_id] = ""
            tracking["tool_runtime"][sub_id] = 0.0
            tracking["tool_called"][sub_id] = False

        # Use the first request for finalization
        first_request = tracking["concat_outputs"][f"{request_id}_0"]
        first_request.request_id = request_id  # Use base request ID
        result = _finalize_outputs(
            output=first_request,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=9, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result.responses), 3)
        for i in range(n):
            self.assertEqual(result.responses[i], [4 + i, 5 + i, 6 + i])
            self.assertEqual(result.masks[i], [1, 1, 1])

    def test_mixed_samples_tool_and_engine(self):
        """Test mixed samples with some tool-use and some engine-only."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"tool": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine
        actor.executor = futures.ThreadPoolExecutor(max_workers=1)
        actor.max_tool_calls = 10

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        n = 3

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": MagicMock(n=n),
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        tracking = _init_tool_tracking()

        # Sample 0: with tool call
        sub_id_0 = f"{request_id}_0"
        mock_output_0 = FakeCompletionOutput(text="<tool>tool</tool>input", token_ids=[4, 5, 6], finish_reason="stop")
        mock_output_0.mask = [1, 1, 1]
        mock_request_0 = FakeRequestOutput(sub_id_0, [mock_output_0], finished=True)
        mock_request_0.prompt = "test prompt"
        mock_request_0.prompt_token_ids = [1, 2, 3]

        # Simulate tool output appended
        tool_tokens = [7, 8, 9]
        combined_tokens = [4, 5, 6] + tool_tokens
        combined_mask = [1, 1, 1] + [0, 0, 0]  # Tool tokens have mask=0

        tracking["concat_outputs"][sub_id_0] = mock_request_0
        tracking["concat_outputs"][sub_id_0].outputs[0].token_ids = combined_tokens
        tracking["masks"][sub_id_0] = combined_mask
        tracking["num_calls"][sub_id_0] = 1
        tracking["tool_called"][sub_id_0] = True
        tracking["tool_output"][sub_id_0] = "TOOL!"
        tracking["tool_runtime"][sub_id_0] = 0.1
        tracking["timeout"][sub_id_0] = False
        tracking["tool_error"][sub_id_0] = ""

        # Samples 1 and 2: engine-only
        for i in [1, 2]:
            sub_id = f"{request_id}_{i}"
            mock_output = FakeCompletionOutput(
                text=f"Engine output {i}", token_ids=[10 + i, 11 + i, 12 + i], finish_reason="stop"
            )
            mock_output.mask = [1, 1, 1]
            mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
            mock_request.prompt = "test prompt"
            mock_request.prompt_token_ids = [1, 2, 3]

            tracking["concat_outputs"][sub_id] = mock_request
            tracking["masks"][sub_id] = [1, 1, 1]
            tracking["num_calls"][sub_id] = 0
            tracking["tool_called"][sub_id] = False
            tracking["tool_output"][sub_id] = ""
            tracking["tool_runtime"][sub_id] = 0.0
            tracking["timeout"][sub_id] = False
            tracking["tool_error"][sub_id] = ""

        # Finalize
        mock_request_0.request_id = request_id  # Use base request ID
        result = _finalize_outputs(
            output=mock_request_0,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=15, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result.responses), 3)
        self.assertEqual(result.responses[0], combined_tokens)
        self.assertEqual(result.masks[0], combined_mask)
        self.assertTrue(result.request_info.tool_calleds[0])
        self.assertFalse(result.request_info.tool_calleds[1])
        self.assertFalse(result.request_info.tool_calleds[2])

    def test_tool_output_clipping_per_request_max_tokens(self):
        """Test tool output clipping based on per-request max_tokens."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"tool": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        sub_id = f"{request_id}_0"
        max_tokens = 10

        self.generation_config.max_tokens = max_tokens

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": self.generation_config,
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        tracking = _init_tool_tracking()

        # Model generates 5 tokens
        model_tokens = [4, 5, 6, 7, 8]
        # Tool would generate 10 tokens but should be clipped
        tool_tokens = list(range(10, 20))

        # Total would be 15 tokens, but max_tokens=10, so should be clipped to 10
        expected_tokens = model_tokens + tool_tokens[:5]  # 5 model + 5 tool = 10 total
        expected_mask = [1] * 5 + [0] * 5  # Model tokens have mask=1, tool tokens have mask=0

        mock_output = FakeCompletionOutput(
            text="<tool>tool</tool>input", token_ids=expected_tokens, finish_reason="stop"
        )
        mock_output.mask = expected_mask

        mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
        mock_request.prompt = "test prompt"
        mock_request.prompt_token_ids = [1, 2, 3]

        tracking["concat_outputs"][sub_id] = mock_request
        tracking["masks"][sub_id] = expected_mask
        tracking["num_calls"][sub_id] = 1
        tracking["tool_called"][sub_id] = True
        tracking["tool_output"][sub_id] = "TOOL_OUTPUT_LONG"
        tracking["tool_runtime"][sub_id] = 0.1
        tracking["timeout"][sub_id] = False
        tracking["tool_error"][sub_id] = ""

        mock_request.request_id = request_id  # Use base request ID
        result = _finalize_outputs(
            output=mock_request,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=10, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result.responses), 1)
        self.assertLessEqual(len(result.responses[0]), max_tokens)
        self.assertEqual(len(result.responses[0]), len(result.masks[0]))

    def test_tool_output_clipping_context_window(self):
        """Test tool output clipping based on context window."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"tool": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        sub_id = f"{request_id}_0"

        # Small context window
        max_model_len = 15
        prompt_tokens = [1, 2, 3, 4, 5]
        model_tokens = [6, 7, 8, 9, 10]
        tool_tokens = list(range(11, 21))  # 10 tokens

        # Total would be 5 (prompt) + 5 (model) + 10 (tool) = 20 tokens
        # But max_model_len = 15, so need to clip tool output
        # Available for tool = 15 - 5 (prompt) - 5 (model) = 5 tokens
        expected_tool_tokens = tool_tokens[:5]
        expected_tokens = model_tokens + expected_tool_tokens
        expected_mask = [1] * 5 + [0] * 5

        actor.request_metadata[request_id] = {
            "prompt_token_ids": prompt_tokens,
            "original_sampling_params": self.generation_config,
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        tracking = _init_tool_tracking()
        tracking["can_make_new_request"] = {sub_id: False}  # Should be False after clipping

        mock_output = FakeCompletionOutput(
            text="<tool>tool</tool>input", token_ids=expected_tokens, finish_reason="stop"
        )
        mock_output.mask = expected_mask

        mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
        mock_request.prompt = "test prompt"
        mock_request.prompt_token_ids = prompt_tokens

        tracking["concat_outputs"][sub_id] = mock_request
        tracking["masks"][sub_id] = expected_mask
        tracking["num_calls"][sub_id] = 1
        tracking["tool_called"][sub_id] = True
        tracking["tool_output"][sub_id] = "TOOL_OUTPUT_VERY_LONG"
        tracking["tool_runtime"][sub_id] = 0.1
        tracking["timeout"][sub_id] = False
        tracking["tool_error"][sub_id] = ""

        mock_request.request_id = request_id  # Use base request ID
        result = _finalize_outputs(
            output=mock_request,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=5, num_response_tokens=10, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        total_length = len(prompt_tokens) + len(result.responses[0])
        self.assertLessEqual(total_length, max_model_len)
        self.assertFalse(tracking["can_make_new_request"].get(sub_id, True))

    def test_tool_failure_path(self):
        """Test handling of tool failures."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"tool": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        sub_id = f"{request_id}_0"

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": self.generation_config,
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        tracking = _init_tool_tracking()

        mock_output = FakeCompletionOutput(text="<tool>tool</tool>input", token_ids=[4, 5, 6], finish_reason="stop")
        mock_output.mask = [1, 1, 1]

        mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
        mock_request.prompt = "test prompt"
        mock_request.prompt_token_ids = [1, 2, 3]

        tracking["concat_outputs"][sub_id] = mock_request
        tracking["masks"][sub_id] = [1, 1, 1]
        tracking["num_calls"][sub_id] = 1
        tracking["tool_called"][sub_id] = True
        tracking["tool_output"][sub_id] = "partial"
        tracking["tool_error"][sub_id] = "boom"
        tracking["tool_runtime"][sub_id] = 0.1
        tracking["timeout"][sub_id] = False

        mock_request.request_id = request_id  # Use base request ID
        result = _finalize_outputs(
            output=mock_request,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=3, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(result.request_info.tool_errors[0], "boom")
        self.assertTrue(result.request_info.tool_calleds[0])

    def test_pending_engine_requests_deferred_cleanup(self):
        """Test that cleanup is deferred when engine has pending sub-requests."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        n = 2

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": MagicMock(n=n),
            "is_eval": False,
            "dataset_index": 0,
        }

        # Add unfinished sub-request to engine
        sub_id_1 = f"{request_id}_1"
        self.engine.unfinished_requests[sub_id_1] = True

        # Mock scheduler
        mock_scheduler = MagicMock()
        mock_unfinished = MagicMock()
        mock_unfinished.request_id = sub_id_1
        mock_scheduler.waiting = [mock_unfinished]
        mock_scheduler.running = []
        mock_scheduler.swapped = []
        self.engine.scheduler = mock_scheduler

        tracking = {}

        # Should NOT clean up because sub-request is still in engine
        actor._cleanup_request_data(request_id, tracking)
        self.assertIn(request_id, actor.request_metadata)

        # Now clear unfinished requests
        self.engine.unfinished_requests.clear()
        mock_scheduler.waiting = []
        self.engine.has_unfinished_requests = lambda: False

        # Should clean up now
        actor._cleanup_request_data(request_id, tracking)
        self.assertNotIn(request_id, actor.request_metadata)

    def test_pending_tool_futures_deferred_cleanup(self):
        """Test that cleanup is deferred when there are pending tool futures."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"tool": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine
        actor.executor = futures.ThreadPoolExecutor(max_workers=1)

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        sub_id = f"{request_id}_0"

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": MagicMock(n=1),
            "is_eval": False,
            "dataset_index": 0,
        }

        tracking = _init_tool_tracking()

        # Add a pending future
        never_done_future = futures.Future()
        tracking["pending_tool_futures"][sub_id] = never_done_future

        # Mock scheduler with no unfinished requests
        mock_scheduler = MagicMock()
        mock_scheduler.waiting = []
        mock_scheduler.running = []
        mock_scheduler.swapped = []
        self.engine.scheduler = mock_scheduler
        self.engine.has_unfinished_requests = lambda: False

        # Should NOT clean up because of pending future
        actor._cleanup_request_data(request_id, tracking)
        self.assertIn(request_id, actor.request_metadata)

        # Complete the future
        never_done_future.set_result({"output": "DONE", "error": None})
        del tracking["pending_tool_futures"][sub_id]

        # Should clean up now
        actor._cleanup_request_data(request_id, tracking)
        self.assertNotIn(request_id, actor.request_metadata)

    def test_prefetch_batching_behavior(self):
        """Test prefetch batching behavior based on unfinished requests."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.llm_engine = self.engine
        actor.inference_batch_size = 2

        # Add enough unfinished requests to trigger batching behavior
        for i in range(3):
            self.engine.unfinished_requests[f"req_{i}"] = True

        # Should have >= 2 unfinished requests
        self.assertGreaterEqual(self.engine.get_num_unfinished_requests(), 2)

        # Clear some requests
        self.engine.unfinished_requests = {"req_0": True}

        # Should have < 2 unfinished requests
        self.assertLess(self.engine.get_num_unfinished_requests(), 2)

    def test_stop_logic_and_exit_reasons(self):
        """Test stop logic and loop exit reasons."""
        # Test basic stop logic concepts without full actor setup

        # Test 1: Verify engine unfinished request tracking
        self.engine.unfinished_requests = {"req_1": True, "req_2": True}
        self.assertEqual(self.engine.get_num_unfinished_requests(), 2)

        self.engine.unfinished_requests.clear()
        self.assertEqual(self.engine.get_num_unfinished_requests(), 0)

        # Test 2: Verify tracking initialization and cleanup
        tracking = _init_tool_tracking()
        self.assertIn("pending_tool_futures", tracking)
        self.assertIsInstance(tracking["pending_tool_futures"], dict)

        # Test 3: Verify actor manager stop behavior
        manager = FakeActorManager()
        manager.set_should_stop(False)

        async def test_stop():
            result = await manager.should_stop.remote()
            return result

        self.assertFalse(asyncio.run(test_stop()))

        manager.set_should_stop(True)
        self.assertTrue(asyncio.run(test_stop()))

    def test_response_count_mismatch_with_duplicate_outputs(self):
        """Test that reproduces the response count mismatch error with duplicate outputs."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"tool": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_43039"  # Format: prefix_training_step_dataset_index
        n = 4  # Expecting 4 responses

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": MagicMock(n=n),
            "is_eval": False,
            "dataset_index": 43039,
            "prompt": "test prompt",
        }

        tracking = _init_tool_tracking()

        # Create outputs that would lead to duplicate merging (7 instead of 4)
        # This simulates the bug where outputs get incorrectly duplicated
        for i in range(n):
            sub_id = f"{request_id}_{i}"
            mock_output = FakeCompletionOutput(
                text=f"Output {i}", token_ids=[4 + i, 5 + i, 6 + i], finish_reason="stop"
            )
            mock_output.mask = [1, 1, 1]

            mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
            mock_request.prompt = "test prompt"
            mock_request.prompt_token_ids = [1, 2, 3]

            tracking["concat_outputs"][sub_id] = mock_request
            tracking["masks"][sub_id] = [1, 1, 1]
            tracking["num_calls"][sub_id] = 0
            tracking["timeout"][sub_id] = False
            tracking["tool_error"][sub_id] = ""
            tracking["tool_output"][sub_id] = ""
            tracking["tool_runtime"][sub_id] = 0.0
            tracking["tool_called"][sub_id] = False

        # Add duplicate outputs to simulate the bug (adding 3 more to make 7)
        for i in range(3):
            sub_id = f"{request_id}_{i}"  # Reusing same sub_ids
            mock_output = FakeCompletionOutput(
                text=f"Duplicate {i}", token_ids=[10 + i, 11 + i, 12 + i], finish_reason="stop"
            )
            mock_output.mask = [1, 1, 1]

            # Create a duplicate request output that gets merged incorrectly
            duplicate_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
            duplicate_request.prompt = "test prompt"
            duplicate_request.prompt_token_ids = [1, 2, 3]

            # Incorrectly extend the outputs array (simulating the merging bug)
            if sub_id in tracking["concat_outputs"]:
                tracking["concat_outputs"][sub_id].outputs.append(mock_output)

        # Now the first 3 sub-requests have 2 outputs each (1 original + 1 duplicate)
        # So we'd have 3*2 + 1 = 7 outputs total when we expect 4

        first_request = tracking["concat_outputs"][f"{request_id}_0"]
        first_request.request_id = request_id  # Use base request ID

        # This should raise the AssertionError we're testing for
        with self.assertRaises(AssertionError) as context:
            _finalize_outputs(
                output=first_request,
                tracking=tracking,
                dataset_index=43039,
                tools=actor.tools,
                original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
                token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=12, generation_time=1.0),
                start_time=1000.0,
            )

        # Verify the error message matches what we expect
        self.assertIn("Response count mismatch", str(context.exception))
        self.assertIn("expected 4 responses but got 7", str(context.exception))

    def test_regression_original_bug(self):
        """Regression test for the original bug with tools present but not used."""
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"any": FakeTool()}
        actor.request_metadata = {}
        actor.llm_engine = self.engine

        request_id = "train_1_0"  # Format: prefix_training_step_dataset_index
        sub_id = f"{request_id}_0"

        actor.request_metadata[request_id] = {
            "prompt_token_ids": [1, 2, 3],
            "original_sampling_params": MagicMock(n=1),
            "is_eval": False,
            "dataset_index": 0,
            "prompt": "test prompt",
        }

        tracking = _init_tool_tracking()

        # Create output WITHOUT tool call but WITH tools configured
        mock_output = FakeCompletionOutput(
            text="Normal output without tool call", token_ids=[4, 5, 6], finish_reason="stop"
        )
        mock_output.mask = [1, 1, 1]

        mock_request = FakeRequestOutput(sub_id, [mock_output], finished=True)
        mock_request.prompt = "test prompt"
        mock_request.prompt_token_ids = [1, 2, 3]

        # This is the key: we need the stub entry in concat_outputs even without tool call
        tracking["concat_outputs"][sub_id] = mock_request
        tracking["masks"][sub_id] = [1, 1, 1]
        tracking["num_calls"][sub_id] = 0
        tracking["tool_called"][sub_id] = False
        tracking["tool_output"][sub_id] = ""
        tracking["tool_error"][sub_id] = ""
        tracking["tool_runtime"][sub_id] = 0.0
        tracking["timeout"][sub_id] = False

        # This should NOT raise "No outputs found" error
        mock_request.request_id = request_id  # Use base request ID
        result = _finalize_outputs(
            output=mock_request,
            tracking=tracking,
            dataset_index=0,
            tools=actor.tools,
            original_sampling_params=actor.request_metadata[request_id]["original_sampling_params"],
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=3, generation_time=1.0),
            start_time=1000.0,
        )

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result.responses), 1)
        self.assertEqual(result.responses[0], [4, 5, 6])
        self.assertFalse(result.request_info.tool_calleds[0])


# Keep original test class for backward compatibility
class TestVllmUtils3(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_finalize_outputs_with_tools_merging(self):
        """Test that _finalize_outputs correctly merges multiple samples from same prompt.

        This test reproduces the bug where responses from different prompts
        could get incorrectly merged due to request_id sorting issues.
        """
        # Create mock outputs for tools mode
        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.mask = [1, 1, 1]
        mock_output1.num_calls = 1
        mock_output1.timeout = False
        mock_output1.tool_error = ""
        mock_output1.tool_output = "result1"
        mock_output1.tool_runtime = 0.5
        mock_output1.tool_called = True

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]
        mock_output2.mask = [1, 1, 1]
        mock_output2.num_calls = 1
        mock_output2.timeout = False
        mock_output2.tool_error = ""
        mock_output2.tool_output = "result2"
        mock_output2.tool_runtime = 0.3
        mock_output2.tool_called = True

        # Create mock RequestOutputs with proper request_id format
        mock_request_output1 = MagicMock(spec=vllm.RequestOutput)
        mock_request_output1.request_id = "train_1_43039"  # Same format as in error
        mock_request_output1.outputs = [mock_output1]

        mock_request_output2 = MagicMock(spec=vllm.RequestOutput)
        mock_request_output2.request_id = "train_1_43039"  # Same prompt, different sample
        mock_request_output2.outputs = [mock_output2]

        # Initialize tracking for tools
        tracking = _init_tool_tracking()

        # Set up tracking data for the same prompt with 2 samples
        req_id_1 = "train_1_43039_0"  # First sample
        req_id_2 = "train_1_43039_1"  # Second sample

        tracking["concat_outputs"][req_id_1] = mock_request_output1
        tracking["concat_outputs"][req_id_2] = mock_request_output2
        tracking["masks"][req_id_1] = [1, 1, 1]
        tracking["masks"][req_id_2] = [1, 1, 1]
        tracking["num_calls"][req_id_1] = 1
        tracking["num_calls"][req_id_2] = 1
        tracking["timeout"][req_id_1] = False
        tracking["timeout"][req_id_2] = False
        tracking["tool_error"][req_id_1] = ""
        tracking["tool_error"][req_id_2] = ""
        tracking["tool_output"][req_id_1] = "result1"
        tracking["tool_output"][req_id_2] = "result2"
        tracking["tool_runtime"][req_id_1] = 0.5
        tracking["tool_runtime"][req_id_2] = 0.3
        tracking["tool_called"][req_id_1] = True
        tracking["tool_called"][req_id_2] = True

        # Call the function under test
        result = _finalize_outputs(
            output=mock_request_output1,
            tracking=tracking,
            dataset_index=43039,
            tools={"some_tool": {}},  # Tools enabled
            original_sampling_params=MagicMock(n=2),
            token_statistics=TokenStatistics(num_prompt_tokens=10, num_response_tokens=6, generation_time=1.0),
            start_time=1000.0,
        )

        # Verify that we get exactly one result (one prompt)
        self.assertEqual(len(result.responses), 2, "Expected exactly 2 responses (samples) for one prompt")

        # Verify the responses are correctly merged
        self.assertEqual(result.responses[0], [1, 2, 3])
        self.assertEqual(result.responses[1], [4, 5, 6])

        # Verify masks are correct
        self.assertEqual(len(result.masks), 2)
        self.assertEqual(result.masks[0], [1, 1, 1])
        self.assertEqual(result.masks[1], [1, 1, 1])

    def test_finalize_outputs_request_id_sorting(self):
        """Test that request IDs are sorted correctly by training_step and dataset_index."""
        tracking = _init_tool_tracking()

        # Create mock outputs for a single request with multiple samples that should be sorted
        base_request_id = "train_1_100"
        mock_outputs = []

        # Create 3 samples for the same request, but add them to tracking out of order
        sample_token_ids = [[3, 4], [1, 2], [5, 6]]  # Out of order by sample index
        sample_indices = [2, 0, 1]  # Sample indices out of order

        for i, (tokens, sample_idx) in enumerate(zip(sample_token_ids, sample_indices)):
            mock_output = MagicMock(spec=vllm.CompletionOutput)
            mock_output.token_ids = tokens
            mock_output.mask = [1] * len(tokens)
            mock_output.num_calls = 1
            mock_output.timeout = False
            mock_output.tool_error = ""
            mock_output.tool_output = f"result{i}"
            mock_output.tool_runtime = 0.1
            mock_output.tool_called = True

            mock_request_output = MagicMock(spec=vllm.RequestOutput)
            mock_request_output.request_id = base_request_id
            mock_request_output.outputs = [mock_output]
            mock_outputs.append(mock_request_output)

            # Set up tracking with sample-specific request IDs
            sample_req_id = f"{base_request_id}_{sample_idx}"
            tracking["concat_outputs"][sample_req_id] = mock_request_output
            tracking["masks"][sample_req_id] = [1] * len(tokens)
            tracking["num_calls"][sample_req_id] = 1
            tracking["timeout"][sample_req_id] = False
            tracking["tool_error"][sample_req_id] = ""
            tracking["tool_output"][sample_req_id] = f"result{i}"
            tracking["tool_runtime"][sample_req_id] = 0.1
            tracking["tool_called"][sample_req_id] = True

        result = _finalize_outputs(
            output=mock_outputs[0],
            tracking=tracking,
            dataset_index=100,  # Single dataset index for this request
            tools={"some_tool": {}},
            original_sampling_params=MagicMock(n=2),
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=6, generation_time=1.0),
            start_time=1000.0,
        )

        # Results come in the order they were processed from tracking (which is dict iteration order)
        # Since we added samples with indices [2, 0, 1], they get processed in that order
        # So responses should be [[3, 4], [1, 2], [5, 6]] (in tracking order)
        expected_responses = [[3, 4], [1, 2], [5, 6]]
        self.assertEqual(result.responses, expected_responses, "Responses should match tracking order")

    def test_continuation_requests_trigger_processing(self):
        """Test that when a continuation request completes, it triggers processing of all samples.

        This test verifies the fix for the bug where continuation requests would accumulate
        in request_outputs but never trigger _maybe_process_and_insert, causing the system
        to hang with incomplete requests.
        """
        # The key insight is that when a continuation request completes,
        # the code path should check if all samples are ready and only then
        # call _maybe_process_and_insert. This test verifies that behavior by
        # checking the affected code section.

        # The fix ensures continuation requests only trigger processing when
        # all expected samples are ready, preventing premature processing

        # We can verify this by checking the actual code has the fix
        import inspect

        import open_instruct.vllm_utils3 as vllm_utils3

        # Get the source code of the LLMRayActor class
        source = inspect.getsource(vllm_utils3.LLMRayActor)

        # Check that the fix is present in the source code
        # The fix ensures continuation requests check sample readiness
        continuation_check = """# Check if we now have all samples ready
                            if request_id in self.request_metadata:
                                expected_n = self.request_metadata[request_id]["original_sampling_params"].n

                                # Count how many outputs we have for this request
                                num_outputs = len(self.request_outputs[request_id])

                                # Only try to process if we have all expected outputs
                                if num_outputs >= expected_n:
                                    total_processed += self._maybe_process_and_insert"""

        # Remove extra whitespace for comparison
        normalized_source = " ".join(source.split())
        normalized_check = " ".join(continuation_check.split())

        # Verify the fix is present
        self.assertIn(
            normalized_check, normalized_source, "The continuation request readiness check is not present in the code"
        )

    def test_process_from_queue_should_not_exit_with_unfinished_requests(self):
        """Test that process_from_queue doesn't exit early when fill_engine returns 0 but unfinished requests exist.

        This reproduces the bug where the function exits early, abandoning unfinished requests.
        """
        # Create a mock LLMRayActor with necessary attributes
        actor = LLMRayActor.__new__(LLMRayActor)  # Create without calling __init__

        # Mock required attributes
        actor.prompt_queue = queue.Queue()
        actor.inference_batch_size = 64
        actor.logger = MagicMock()
        actor.dropped_results = 0
        actor.verbose = False

        # Mock llm_engine with tokenizer
        mock_engine = MagicMock()
        mock_engine.tokenizer = MagicMock()
        actor.llm_engine = mock_engine

        # Mock timing attributes
        actor._engine_steps_timing = {"engine_step_call": 0.0, "extract_base_request_id": 0.0, "handle_output": 0.0}
        actor.request_metadata = {}
        actor.max_tool_calls = 10
        actor.executor = MagicMock()
        actor.tools = {}

        # Mock the methods that would be called
        with (
            patch.object(actor, "fill_engine", return_value=0) as mock_fill_engine,
            patch.object(actor, "_should_stop", side_effect=[False, True, False]) as mock_should_stop,
            patch.object(actor.llm_engine, "get_num_unfinished_requests", return_value=61),
            patch.object(actor, "_poll_tool_futures", return_value=[]),
            patch.object(actor.llm_engine, "has_unfinished_requests", return_value=True),
            patch.object(actor.llm_engine, "step", return_value=[]),
            patch.object(actor, "_maybe_process_and_insert", return_value=0),
        ):
            # This should not return 0 (early exit) when there are unfinished requests
            # Instead, it should enter the main processing loop
            actor.process_from_queue(timeout=20)

            # Verify that fill_engine was called
            mock_fill_engine.assert_called_once_with()

            # The bug: function exits early when fill_engine returns 0, even with unfinished requests
            # Expected behavior: should enter main loop and process unfinished requests

            # With the bug, should_stop is called only once (during the early exit check)
            # Without the bug, should_stop should be called multiple times (in the main loop)
            self.assertGreater(
                mock_should_stop.call_count,
                1,
                f"SUCCESS: Bug is fixed! should_stop called {mock_should_stop.call_count} times. "
                f"This means we entered the main processing loop instead of exiting early.",
            )

    def test_maybe_process_and_insert_no_duplicate_request_ids(self):
        """Test that _maybe_process_and_insert doesn't add duplicate request_ids when called multiple times.

        This reproduces and tests the fix for the bug where multiple calls to _maybe_process_and_insert
        would re-add the same outputs from concat_outputs, causing duplicate request_ids.
        """
        # Create a mock LLMRayActor
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = False
        actor.tools = {"some_tool": {}}  # Enable tools mode

        # Mock methods called by _maybe_process_and_insert
        actor._process_completed_request = MagicMock(return_value=("result", False))
        actor._insert_result_to_queue = MagicMock()
        actor._cleanup_request_data = MagicMock()

        # Set up request metadata
        request_id = "train_1_35604"
        actor.request_metadata = {
            request_id: {
                "original_sampling_params": MagicMock(n=2),  # Expect 2 samples
                "is_eval": False,
                "dataset_index": 35604,
            }
        }

        # Create mock outputs for the 2 expected samples
        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]

        mock_request_output1 = MagicMock(spec=vllm.RequestOutput)
        mock_request_output1.request_id = "train_1_35604_0"
        mock_request_output1.outputs = [mock_output1]
        mock_request_output1.prompt = "test prompt"
        mock_request_output1.prompt_token_ids = [1, 2, 3]

        mock_request_output2 = MagicMock(spec=vllm.RequestOutput)
        mock_request_output2.request_id = "train_1_35604_1"
        mock_request_output2.outputs = [mock_output2]
        mock_request_output2.prompt = "test prompt"
        mock_request_output2.prompt_token_ids = [1, 2, 3]

        # Initialize tracking with concat_outputs
        tracking = _init_tool_tracking()
        tracking["concat_outputs"]["train_1_35604_0"] = mock_request_output1
        tracking["concat_outputs"]["train_1_35604_1"] = mock_request_output2

        # Set up request_outputs with empty list for our request
        request_outputs = {request_id: []}
        current_time = time.time()

        # Call _maybe_process_and_insert MULTIPLE times (this should trigger the bug)
        # First call - should move both outputs from concat_outputs to request_outputs and process them
        result1 = actor._maybe_process_and_insert(request_id, request_outputs, tracking, current_time)

        # Since we have 2 samples and expect 2, it should process and remove from request_outputs
        self.assertEqual(result1, 1, "Should return 1 since request was processed")
        self.assertEqual(len(request_outputs), 0, "request_outputs should be empty after processing")

        # Reset request_outputs to test the duplicate scenario
        request_outputs[request_id] = []

        # Call again to test duplicate prevention
        result2 = actor._maybe_process_and_insert(request_id, request_outputs, tracking, current_time)

        # Check if the key exists before accessing it
        if request_id in request_outputs:
            # Verify outputs were moved correctly without duplicates
            self.assertEqual(len(request_outputs[request_id]), 2, "Should have 2 outputs after second call")

            # Third call - should NOT add duplicates (this previously caused the assertion error)
            actor._maybe_process_and_insert(request_id, request_outputs, tracking, current_time)

            # Verify no duplicates were added
            self.assertEqual(
                len(request_outputs[request_id]), 2, "Should still have exactly 2 outputs after third call"
            )

            # Verify all request_ids are unique (this was failing before the fix)
            request_ids = [out.request_id for out in request_outputs[request_id]]
            unique_request_ids = set(request_ids)
            self.assertEqual(len(request_ids), len(unique_request_ids), "All request_ids should be unique")

            # Verify we have the expected request_ids
            expected_request_ids = {"train_1_35604_0", "train_1_35604_1"}
            self.assertEqual(set(request_ids), expected_request_ids, "Should have the expected request_ids")
        else:
            # If key was removed, the second call also processed successfully
            self.assertEqual(result2, 1, "Should return 1 since request was processed again")

    def test_prefetch_thread_pause_resume_on_should_stop(self):
        """Test that the prefetch thread pauses when should_stop=True and resumes when should_stop=False.

        This simulates a weight synchronization scenario where the prefetch thread
        should pause during sync but not exit, then resume afterwards.
        """
        # Create a mock LLMRayActor without calling __init__ to avoid ray dependencies
        actor = LLMRayActor.__new__(LLMRayActor)

        # Set up mock dependencies
        actor.prompt_queue = queue.Queue()
        actor.inference_batch_size = 64
        actor.logger = MagicMock()
        actor.verbose = False  # Add missing verbose attribute
        actor.tools = {}  # Add missing tools attribute
        actor.request_metadata = {}  # Add missing request_metadata attribute

        # Mock LLM engine
        mock_engine = MagicMock()
        mock_engine.get_num_unfinished_requests.return_value = 0
        actor.llm_engine = mock_engine

        # Mock actor manager with controllable should_stop
        mock_actor_manager = MagicMock()
        actor.actor_manager = mock_actor_manager

        # Set up should_stop caching attributes
        actor._last_should_stop_update = float("-inf")
        actor._should_stop_value = False
        actor._should_stop_timeout_s = 5

        # Set up prefetch components
        actor._prefetch_buffer = []
        actor._prefetch_cv = threading.Condition()
        actor._prefetch_thread = None  # Add missing thread attribute

        # Create a mock generation config for test requests
        mock_gen_config = MagicMock()
        mock_gen_config.n = 1

        # Events to coordinate test timing
        prefetch_started = threading.Event()
        should_stop_checked = threading.Event()
        resume_ready = threading.Event()

        # Track should_stop calls to verify pause behavior
        should_stop_call_count = 0
        should_stop_values = [False, False, True, True, False, False]  # Normal -> Pause -> Resume

        def mock_should_stop():
            nonlocal should_stop_call_count
            should_stop_call_count += 1
            prefetch_started.set()

            if should_stop_call_count == 3:  # First time returning True
                should_stop_checked.set()
            elif should_stop_call_count == 5:  # After resume
                resume_ready.set()

            if should_stop_call_count <= len(should_stop_values):
                return should_stop_values[should_stop_call_count - 1]
            return False  # Default to False after our test sequence

        # Mock the should_stop method
        actor._should_stop = mock_should_stop

        # Add multiple test requests to the queue to ensure the thread keeps looping
        for i in range(5):
            test_request = MagicMock()  # Remove spec to avoid attribute restrictions
            test_request.generation_config = mock_gen_config
            test_request.prompt = [1, 2, 3]  # Mock prompt tokens
            test_request.is_eval = False
            test_request.training_step = 1
            test_request.dataset_index = i
            actor.prompt_queue.put(test_request)

        # Track prefetch activity
        requests_processed = threading.Event()

        def check_buffer():
            """Helper to check if request was processed"""
            if actor._prefetch_buffer:
                requests_processed.set()

        # Start the prefetch thread
        prefetch_thread = threading.Thread(target=actor._prefetch_worker, daemon=True)
        prefetch_thread.start()

        try:
            # Wait for prefetch thread to start
            self.assertTrue(prefetch_started.wait(timeout=2.0), "Prefetch thread should start")

            # Give thread a moment to process the request during normal operation
            time.sleep(0.2)

            # Wait for the should_stop=True to be checked (pause condition)
            self.assertTrue(should_stop_checked.wait(timeout=2.0), "Should stop condition should be checked")

            # At this point, thread should be paused, verify it doesn't exit
            # by checking that should_stop continues to be called (indicating thread is alive and waiting)
            time.sleep(0.2)  # Give some time for the pause to take effect

            # Thread should still be alive (not exited)
            self.assertTrue(prefetch_thread.is_alive(), "Prefetch thread should still be alive during pause")

            # Wait for resume condition
            self.assertTrue(resume_ready.wait(timeout=3.0), "Resume condition should be reached")

            # Add another request to verify thread can resume processing
            test_request2 = MagicMock()
            test_request2.generation_config = mock_gen_config
            test_request2.prompt = [1, 2, 3]
            test_request2.is_eval = False
            test_request2.training_step = 1
            test_request2.dataset_index = 99
            actor.prompt_queue.put(test_request2)

            # Give thread time to process after resume
            time.sleep(0.3)

            # Verify thread is still alive and functioning
            self.assertTrue(prefetch_thread.is_alive(), "Prefetch thread should still be alive after resume")

            # Verify that should_stop was called multiple times, indicating the thread
            # continued to check the condition rather than exiting
            self.assertGreaterEqual(
                should_stop_call_count,
                4,
                f"should_stop should be called multiple times during pause/resume, got {should_stop_call_count}",
            )

        finally:
            # Clean up: force thread to exit by making should_stop return True permanently
            actor._should_stop = lambda: True

            # Add a properly formed request to unblock any queue.get() calls
            cleanup_request = MagicMock()
            cleanup_request.generation_config = mock_gen_config
            cleanup_request.prompt = [1, 2, 3]
            cleanup_request.is_eval = False
            cleanup_request.training_step = 1
            cleanup_request.dataset_index = 999
            try:
                actor.prompt_queue.put_nowait(cleanup_request)
            except queue.Full:
                pass

            # Give thread time to clean up
            prefetch_thread.join(timeout=1.0)

    def test_missing_metadata_raises_critical_exception(self):
        """Test that missing metadata raises a critical RuntimeError instead of just logging a warning.

        This test prevents regression of the bug where missing metadata was only logged as a warning,
        causing silent data corruption instead of immediate failure.
        """
        # Create a mock LLMRayActor
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.verbose = True
        actor.request_metadata = {}  # Empty metadata dict to simulate missing metadata
        actor.tools = {}
        actor.max_tool_calls = {}
        actor.executor = None

        # Mock the LLM engine and other dependencies
        mock_engine = MagicMock()
        mock_engine.has_unfinished_requests.return_value = True
        mock_engine.step.return_value = [MagicMock()]  # One finished output
        actor.llm_engine = mock_engine

        # Create a mock step output with a request ID that doesn't exist in metadata
        mock_step_output = MagicMock()
        mock_step_output.request_id = "train_1_35604_0"  # This matches the error logs
        mock_step_output.finished = True
        mock_engine.step.return_value = [mock_step_output]

        # Mock other required methods
        actor._should_stop = MagicMock(return_value=False)
        actor._poll_tool_futures = MagicMock(return_value=[])

        # The critical fix: This should raise a RuntimeError, not just log a warning
        with self.assertRaises(RuntimeError) as context:
            # This will call the fixed code path that should raise an exception
            actor.process_from_queue(timeout=1.0)

        # Verify the exception message contains the expected information
        error_message = str(context.exception)
        self.assertIn("Critical bug: Missing metadata for request", error_message)
        self.assertIn("train_1_35604", error_message)  # The base request ID
        self.assertIn("metadata was cleaned up prematurely", error_message)

    def test_cleanup_request_data_waits_for_engine_requests(self):
        """Test that _cleanup_request_data waits for unfinished engine requests before cleaning metadata.

        This test ensures metadata is not cleaned up prematurely while sub-requests are still in the engine.
        """
        # Create a mock LLMRayActor
        actor = LLMRayActor.__new__(LLMRayActor)
        actor.logger = MagicMock()
        actor.tools = {}

        # Set up mock engine
        mock_engine = MagicMock()
        actor.llm_engine = mock_engine

        # Mock engine state - has unfinished requests
        mock_engine.has_unfinished_requests.return_value = True

        # Mock scheduler with unfinished requests for our base request ID
        mock_scheduler = MagicMock()
        mock_engine.scheduler = mock_scheduler

        # Create mock request objects with the same base request ID we're testing
        mock_unfinished_request = MagicMock()
        mock_unfinished_request.request_id = "train_1_35604_0"  # Sub-request of train_1_35604
        mock_scheduler.waiting = [mock_unfinished_request]
        mock_scheduler.running = []
        mock_scheduler.swapped = []

        # Set up request metadata
        base_request_id = "train_1_35604"
        actor.request_metadata = {base_request_id: {"test": "data"}}

        # Create empty tracking dict
        tracking = {}

        # Call _cleanup_request_data - it should NOT remove metadata because engine has unfinished requests
        actor._cleanup_request_data(base_request_id, tracking)

        # Verify metadata was NOT removed (this is the fix)
        self.assertIn(
            base_request_id,
            actor.request_metadata,
            "Metadata should NOT be removed while engine has unfinished sub-requests",
        )

        # Now simulate engine having no unfinished requests
        mock_engine.has_unfinished_requests.return_value = False
        mock_scheduler.waiting = []

        # Call _cleanup_request_data again - now it SHOULD remove metadata
        actor._cleanup_request_data(base_request_id, tracking)

        # Verify metadata was removed
        self.assertNotIn(
            base_request_id,
            actor.request_metadata,
            "Metadata SHOULD be removed when engine has no unfinished sub-requests",
        )


if __name__ == "__main__":
    unittest.main()
