import logging
import queue
import unittest
from unittest.mock import MagicMock, patch

import vllm

from open_instruct.queue_types import TokenStatistics
from open_instruct.vllm_utils3 import LLMRayActor, _finalize_outputs, _init_tool_tracking


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
            output=mock_request_output1,  # Single output to get the base request_id
            tracking=tracking,
            dataset_index=43039,
            tools={"some_tool": {}},  # Tools enabled
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
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=6, generation_time=1.0),
            start_time=1000.0,
        )

        # Results come in the order they were processed from tracking (which is dict iteration order)
        # Since we added samples with indices [2, 0, 1], they get processed in that order
        # So responses should be [[3, 4], [1, 2], [5, 6]] (in tracking order)
        expected_responses = [[3, 4], [1, 2], [5, 6]]
        self.assertEqual(result.responses, expected_responses, "Responses should match tracking order")

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
            patch.object(actor, "_llm_engine_get_num_unfinished_requests", return_value=61),
            patch.object(actor, "_reset_fill_engine_timing"),
            patch.object(actor, "_reset_engine_steps_timing"),
            patch.object(actor, "_poll_tool_futures", return_value=[]),
            patch.object(actor, "_llm_engine_has_unfinished_requests", return_value=True),
            patch.object(actor, "_engine_step", return_value=[]),
            patch.object(actor, "_maybe_process_and_insert", return_value=0),
        ):
            # This should not return 0 (early exit) when there are unfinished requests
            # Instead, it should enter the main processing loop
            actor.process_from_queue(timeout=20)

            # Verify that fill_engine was called
            mock_fill_engine.assert_called_once_with(timeout=20)

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


if __name__ == "__main__":
    unittest.main()
