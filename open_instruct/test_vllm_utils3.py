import logging
import unittest
from unittest.mock import MagicMock

import vllm

from open_instruct.queue_types import TokenStatistics
from open_instruct.vllm_utils3 import _finalize_outputs, _init_tool_tracking


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
            outputs=[],  # Not used in tool mode
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

        # Create mock outputs with different dataset indices but same training step
        mock_outputs = []
        request_ids = ["train_1_100", "train_1_50", "train_1_200"]  # Out of order

        for i, req_id in enumerate(request_ids):
            mock_output = MagicMock(spec=vllm.CompletionOutput)
            mock_output.token_ids = [i + 1]
            mock_output.mask = [1]
            mock_output.num_calls = 1
            mock_output.timeout = False
            mock_output.tool_error = ""
            mock_output.tool_output = f"result{i}"
            mock_output.tool_runtime = 0.1
            mock_output.tool_called = True

            mock_request_output = MagicMock(spec=vllm.RequestOutput)
            mock_request_output.request_id = req_id
            mock_request_output.outputs = [mock_output]
            mock_outputs.append(mock_request_output)

            # Set up tracking
            sample_req_id = f"{req_id}_0"
            tracking["concat_outputs"][sample_req_id] = mock_request_output
            tracking["masks"][sample_req_id] = [1]
            tracking["num_calls"][sample_req_id] = 1
            tracking["timeout"][sample_req_id] = False
            tracking["tool_error"][sample_req_id] = ""
            tracking["tool_output"][sample_req_id] = f"result{i}"
            tracking["tool_runtime"][sample_req_id] = 0.1
            tracking["tool_called"][sample_req_id] = True

        result = _finalize_outputs(
            outputs=mock_outputs,
            tracking=tracking,
            dataset_index=[50, 100, 200],  # Should be sorted by index
            tools={"some_tool": {}},
            token_statistics=TokenStatistics(num_prompt_tokens=3, num_response_tokens=3, generation_time=1.0),
            start_time=1000.0,
        )

        # Results should be sorted by dataset index: 50, 100, 200
        # So responses should be [2], [1], [3] (corresponding to original order)
        expected_responses = [[2], [1], [3]]
        self.assertEqual(result.responses, expected_responses, "Responses should be sorted by dataset index")


if __name__ == "__main__":
    unittest.main()
