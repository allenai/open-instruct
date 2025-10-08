import logging
import unittest
from unittest.mock import MagicMock

import vllm

from open_instruct.vllm_utils3 import process_completed_request


class TestVllmUtils3(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_process_outputs_with_tools(self):
        """Test that process_completed_request correctly handles outputs with tool attributes.

        Tests the new process_completed_request function which combined process_output and _process_completed_request.
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
        mock_output1.finish_reason = "stop"

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]
        mock_output2.mask = [1, 1, 1]
        mock_output2.num_calls = 2
        mock_output2.timeout = False
        mock_output2.tool_error = ""
        mock_output2.tool_output = "result2"
        mock_output2.tool_runtime = 0.3
        mock_output2.tool_called = True
        mock_output2.finish_reason = "stop"

        # Create mock RequestOutput with multiple outputs
        mock_request_output = MagicMock(spec=vllm.RequestOutput)
        mock_request_output.request_id = "train_1_43039"
        mock_request_output.outputs = [mock_output1, mock_output2]
        mock_request_output.prompt = "test prompt"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.prompt_logprobs = None
        mock_request_output.finished = True

        # Setup request metadata
        request_metadata = {
            "train_1_43039": {
                "is_eval": False,
                "dataset_index": 43039,
                "training_step": 1,
                "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "start_time": 1000.0,
            }
        }

        # Mock tools dict to enable tool mode
        tools = {"</tool>": MagicMock()}

        # Call the function under test with tools enabled
        result, is_eval = process_completed_request(
            request_id="train_1_43039",
            outs=[mock_request_output],
            tracking={},  # Not used for this test
            current_time=1001.0,
            tools=tools,
            request_metadata=request_metadata,
        )

        # Verify is_eval is correct
        self.assertFalse(is_eval)

        # Verify that we get both responses
        self.assertEqual(len(result.responses), 2, "Expected exactly 2 responses")

        # Verify the responses are correct
        self.assertEqual(result.responses[0], [1, 2, 3])
        self.assertEqual(result.responses[1], [4, 5, 6])

        # Verify masks are correct
        self.assertEqual(len(result.masks), 2)
        self.assertEqual(result.masks[0], [1, 1, 1])
        self.assertEqual(result.masks[1], [1, 1, 1])

        # Verify request_info has correct tool attributes
        self.assertEqual(result.request_info.num_calls, [1, 2])
        self.assertEqual(result.request_info.tool_outputs, ["result1", "result2"])
        self.assertEqual(result.request_info.tool_runtimes, [0.5, 0.3])
        self.assertEqual(result.request_info.tool_calleds, [True, True])

    def test_process_outputs_without_tools(self):
        """Test that process_completed_request correctly handles outputs without tool attributes."""
        # Create mock outputs without tool attributes
        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.finish_reason = "stop"

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]
        mock_output2.finish_reason = "length"

        # Create mock RequestOutput with multiple outputs
        mock_request_output = MagicMock(spec=vllm.RequestOutput)
        mock_request_output.request_id = "eval_2_200"
        mock_request_output.outputs = [mock_output1, mock_output2]
        mock_request_output.prompt = "test prompt"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.prompt_logprobs = None
        mock_request_output.finished = True

        # Setup request metadata
        request_metadata = {
            "eval_2_200": {
                "is_eval": True,
                "dataset_index": 200,
                "training_step": 2,
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "start_time": 2000.0,
            }
        }

        # Call the function under test without tools (None or empty dict)
        result, is_eval = process_completed_request(
            request_id="eval_2_200",
            outs=[mock_request_output],
            tracking={},  # Not used for this test
            current_time=2000.5,
            tools=None,
            request_metadata=request_metadata,
        )

        # Verify is_eval is correct
        self.assertTrue(is_eval)

        # Verify that we get both responses
        self.assertEqual(len(result.responses), 2, "Expected exactly 2 responses")

        # Verify the responses are correct
        self.assertEqual(result.responses[0], [1, 2, 3])
        self.assertEqual(result.responses[1], [4, 5, 6])

        # Verify finish reasons
        self.assertEqual(result.finish_reasons[0], "stop")
        self.assertEqual(result.finish_reasons[1], "length")

        # Verify default masks (all 1s when no tools)
        self.assertEqual(result.masks[0], [1, 1, 1])
        self.assertEqual(result.masks[1], [1, 1, 1])

        # Verify request_info has default values when tools are not used
        self.assertEqual(result.request_info.num_calls, [0, 0])
        self.assertEqual(result.request_info.timeouts, [False, False])
        self.assertEqual(result.request_info.tool_errors, ["", ""])
        self.assertEqual(result.request_info.tool_outputs, ["", ""])
        self.assertEqual(result.request_info.tool_runtimes, [0.0, 0.0])
        self.assertEqual(result.request_info.tool_calleds, [False, False])


if __name__ == "__main__":
    unittest.main()
