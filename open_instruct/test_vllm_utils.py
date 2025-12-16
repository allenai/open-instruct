import logging
import unittest
from unittest.mock import MagicMock

import vllm
from parameterized import parameterized

from open_instruct import vllm_utils
from open_instruct.data_types import PromptRequest


class TestTruncateToolOutputTokens(unittest.TestCase):
    @parameterized.expand(
        [
            ("no_truncation", [1, 2, 3, 4, 5], 10, 5, 100, 50, [1, 2, 3, 4, 5], 0),
            ("truncate_max_model_len", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 5, 100, 50, [1, 2, 3, 4, 5], 5),
            ("truncate_max_tokens", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 10, 47, 100, 50, [1, 2, 3], 0),
            ("truncate_both_limits", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 95, 47, 100, 50, [1, 2, 3], 5),
            ("no_space_for_model", [1, 2, 3], 100, 5, 100, 50, [], 3),
            ("no_remaining_response", [1, 2, 3], 10, 50, 100, 50, [], 0),
            ("empty_input", [], 10, 5, 100, 50, [], 0),
        ]
    )
    def test_truncate_tool_output_tokens(
        self, name, tokens, prompt_len, response_len, max_model_len, max_tokens, expected_tokens, expected_excess
    ):
        result, excess = vllm_utils.truncate_tool_output_tokens(
            tokens, prompt_len, response_len, max_model_len, max_tokens
        )
        self.assertEqual(result, expected_tokens)
        self.assertEqual(excess, expected_excess)


class TestVllmUtils3(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    def test_process_outputs_with_tools(self):
        """Test that process_completed_request correctly handles outputs with tool attributes.

        Tests the new process_completed_request function which combined process_output and _process_completed_request.
        """

        def create_mock_logprobs(token_ids):
            return [-0.1 * tid for tid in token_ids]

        dataset_index = 43039
        epoch = 0
        prompt_id = f"{epoch}_{dataset_index}"

        mock_request = PromptRequest(
            prompt=[1, 2, 3], generation_config=None, is_eval=False, dataset_index=dataset_index, prompt_id=prompt_id
        )
        request_id = vllm_utils.make_request_id(mock_request)

        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.logprobs = create_mock_logprobs([1, 2, 3])
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
        mock_output2.logprobs = create_mock_logprobs([4, 5, 6])
        mock_output2.mask = [1, 1, 1]
        mock_output2.num_calls = 2
        mock_output2.timeout = False
        mock_output2.tool_error = ""
        mock_output2.tool_output = "result2"
        mock_output2.tool_runtime = 0.3
        mock_output2.tool_called = True
        mock_output2.finish_reason = "stop"

        mock_request_output = MagicMock(spec=vllm.RequestOutput)
        mock_request_output.request_id = request_id
        mock_request_output.outputs = [mock_output1, mock_output2]
        mock_request_output.prompt = "test prompt"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.finished = True

        request_metadata = {
            request_id: {
                "is_eval": False,
                "dataset_index": dataset_index,
                "prompt_id": prompt_id,
                "prompt_token_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "start_time": 1000.0,
            }
        }

        tools = {"</tool>": MagicMock()}

        result, is_eval = vllm_utils.process_completed_request(
            request_id=request_id,
            outs=[mock_request_output],
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

        def create_mock_logprobs(token_ids):
            return [-0.1 * tid for tid in token_ids]

        dataset_index = 200
        epoch = 0
        prompt_id = f"{epoch}_{dataset_index}"

        mock_request = PromptRequest(
            prompt=[1, 2, 3], generation_config=None, is_eval=True, dataset_index=dataset_index, prompt_id=prompt_id
        )
        request_id = vllm_utils.make_request_id(mock_request)

        mock_output1 = MagicMock(spec=vllm.CompletionOutput)
        mock_output1.token_ids = [1, 2, 3]
        mock_output1.logprobs = create_mock_logprobs([1, 2, 3])
        mock_output1.finish_reason = "stop"

        mock_output2 = MagicMock(spec=vllm.CompletionOutput)
        mock_output2.token_ids = [4, 5, 6]
        mock_output2.logprobs = create_mock_logprobs([4, 5, 6])
        mock_output2.finish_reason = "length"

        mock_request_output = MagicMock(spec=vllm.RequestOutput)
        mock_request_output.request_id = request_id
        mock_request_output.outputs = [mock_output1, mock_output2]
        mock_request_output.prompt = "test prompt"
        mock_request_output.prompt_token_ids = [1, 2, 3]
        mock_request_output.finished = True

        request_metadata = {
            request_id: {
                "is_eval": True,
                "dataset_index": dataset_index,
                "prompt_id": prompt_id,
                "prompt_token_ids": [1, 2, 3, 4, 5],
                "start_time": 2000.0,
            }
        }

        result, is_eval = vllm_utils.process_completed_request(
            request_id=request_id,
            outs=[mock_request_output],
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
