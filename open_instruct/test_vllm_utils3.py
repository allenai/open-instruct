import logging
import unittest
from unittest.mock import MagicMock

import vllm
from parameterized import parameterized

from open_instruct.queue_types import PromptRequest, TokenStatistics
from open_instruct.vllm_utils3 import _extract_base_request_id, _finalize_outputs, _init_tool_tracking, make_request_id


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

        # Create a mock output and sampling params
        mock_output = MagicMock(spec=vllm.RequestOutput)
        mock_output.request_id = "train_1_43039"
        mock_output.prompt = []
        mock_output.prompt_token_ids = []
        mock_output.prompt_logprobs = None
        mock_output.outputs = [mock_output1, mock_output2]
        mock_output.finished = True

        mock_sampling_params = MagicMock(spec=vllm.SamplingParams)
        mock_sampling_params.n = 2  # Two samples per prompt

        # Call the function under test
        result = _finalize_outputs(
            output=mock_output,
            tracking=tracking,
            dataset_index=43039,
            training_step=1,
            tools={"some_tool": {}},  # Tools enabled
            original_sampling_params=mock_sampling_params,
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

        # Create a mock output and sampling params
        mock_final_output = MagicMock(spec=vllm.RequestOutput)
        mock_final_output.request_id = base_request_id
        mock_final_output.prompt = []
        mock_final_output.prompt_token_ids = []
        mock_final_output.prompt_logprobs = None
        mock_final_output.outputs = [
            mock_outputs[0].outputs[0],
            mock_outputs[1].outputs[0],
            mock_outputs[2].outputs[0],
        ]
        mock_final_output.finished = True

        mock_sampling_params = MagicMock(spec=vllm.SamplingParams)
        mock_sampling_params.n = 3  # Three samples per prompt

        result = _finalize_outputs(
            output=mock_final_output,
            tracking=tracking,
            dataset_index=100,  # Single dataset index for this request
            training_step=1,
            tools={"some_tool": {}},
            original_sampling_params=mock_sampling_params,
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=6, generation_time=1.0),
            start_time=1000.0,
        )

        # Results should be sorted by sample index (0, 1, 2) regardless of insertion order
        # The function now correctly sorts by sample index, not tracking insertion order
        # Sample 0 has tokens [1, 2], sample 1 has tokens [5, 6], sample 2 has tokens [3, 4]
        expected_responses = [[1, 2], [5, 6], [3, 4]]
        self.assertEqual(result.responses, expected_responses, "Responses should be sorted by sample index")

    @parameterized.expand(
        [
            ("train_1_43039_0", "train_1_43039"),
            ("eval_5_12345_2", "eval_5_12345"),
            ("train_0_999_1", "train_0_999"),
            ("eval_123_456_0", "eval_123_456"),
            ("prefix_step_idx_sample", "prefix_step_idx"),
            ("a_b_c_d", "a_b_c"),
        ]
    )
    def test_extract_base_request_id(self, full_request_id, expected_base_id):
        """Test _extract_base_request_id removes sample suffix correctly."""
        result = _extract_base_request_id(full_request_id)
        self.assertEqual(result, expected_base_id)

    @parameterized.expand(
        [
            (False, 0, 0, "train_0_0"),
            (False, 1, 43039, "train_1_43039"),
            (True, 5, 12345, "eval_5_12345"),
            (False, 100, 999, "train_100_999"),
            (True, 123, 456, "eval_123_456"),
        ]
    )
    def test_make_request_id(self, is_eval, training_step, dataset_index, expected_id):
        """Test make_request_id generates correct request IDs."""
        request = PromptRequest(
            prompt=[1, 2, 3],
            generation_config={},
            training_step=training_step,
            dataset_index=dataset_index,
            is_eval=is_eval,
        )
        result = make_request_id(request)
        self.assertEqual(result, expected_id)

    def test_make_request_id_extract_base_request_id_roundtrip(self):
        """Test that make_request_id and _extract_base_request_id work together correctly."""
        request = PromptRequest(
            prompt=[1, 2, 3], generation_config={}, training_step=42, dataset_index=100, is_eval=False
        )

        # Generate the request ID
        request_id = make_request_id(request)
        self.assertEqual(request_id, "train_42_100")

        # Add a sample suffix (as would happen in vLLM processing)
        full_request_id = f"{request_id}_0"

        # Extract base should give us back the original
        extracted = _extract_base_request_id(full_request_id)
        self.assertEqual(extracted, request_id)


if __name__ == "__main__":
    unittest.main()
