"""Tests for tool_vllm module."""

import unittest
from unittest import mock

import torch
from parameterized import parameterized

from open_instruct.tool_utils import tool_vllm


class TestToolOutput(unittest.TestCase):
    """Test the ToolOutput dataclass."""

    def test_tool_output_initialization(self):
        """Test ToolOutput initialization with default values."""
        output = tool_vllm.ToolOutput(output="test output", called=True, error="", timeout=False, runtime=1.5)
        self.assertEqual(output.output, "test output")
        self.assertTrue(output.called)
        self.assertEqual(output.error, "")
        self.assertFalse(output.timeout)
        self.assertEqual(output.runtime, 1.5)
        self.assertEqual(output.start_str, "<output>\n")
        self.assertEqual(output.end_str, "\n</output>")


class TestMaxCallsExceededTool(unittest.TestCase):
    """Test the MaxCallsExceededTool class."""

    def test_max_calls_exceeded_returns_correct_output(self):
        """Test that MaxCallsExceededTool returns the expected output."""
        tool = tool_vllm.MaxCallsExceededTool(start_str="<tool>", end_str="</tool>")
        result = tool("any prompt")

        self.assertEqual(result.output, "Max tool calls exceeded.")
        self.assertFalse(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)


class TestPythonCodeTool(unittest.TestCase):
    """Test the PythonCodeTool class."""

    @mock.patch("open_instruct.tool_utils.tool_vllm.requests.post")
    def test_python_code_tool_successful_execution(self, mock_post):
        """Test successful code execution via API."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"output": "Hello, World!", "error": ""}
        mock_post.return_value = mock_response

        tool = tool_vllm.PythonCodeTool(api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>")

        prompt = "<code>print('Hello, World!')</code>"
        result = tool(prompt)

        self.assertEqual(result.output, "Hello, World!")
        self.assertTrue(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        mock_post.assert_called_once()

    @mock.patch("open_instruct.tool_utils.tool_vllm.requests.post")
    def test_python_code_tool_with_error(self, mock_post):
        """Test code execution that returns an error."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"output": "", "error": "NameError: name 'foo' is not defined"}
        mock_post.return_value = mock_response

        tool = tool_vllm.PythonCodeTool(api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>")

        prompt = "<code>print(foo)</code>"
        result = tool(prompt)

        self.assertIn("NameError", result.output)
        self.assertTrue(result.called)
        self.assertEqual(result.error, "NameError: name 'foo' is not defined")
        self.assertFalse(result.timeout)

    @mock.patch("open_instruct.tool_utils.tool_vllm.requests.post")
    def test_python_code_tool_timeout(self, mock_post):
        """Test code execution timeout."""
        import requests

        mock_post.side_effect = requests.Timeout("Request timed out")

        tool = tool_vllm.PythonCodeTool(api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>")

        prompt = "<code>import time; time.sleep(10)</code>"
        result = tool(prompt)

        self.assertIn("Timeout after", result.output)
        self.assertTrue(result.called)
        self.assertTrue(result.timeout)

    def test_python_code_tool_no_code_blocks(self):
        """Test when no code blocks are found."""
        tool = tool_vllm.PythonCodeTool(api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>")

        prompt = "This prompt has no code blocks"
        result = tool(prompt)

        self.assertEqual(result.output, "")
        self.assertFalse(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)

    @parameterized.expand(
        [
            ("<code>print('first')</code> <code>print('second')</code>", "print('second')"),
            ("<code>a=1</code> some text <code>b=2</code>", "b=2"),
        ]
    )
    @mock.patch("open_instruct.tool_utils.tool_vllm.requests.post")
    def test_python_code_tool_uses_last_code_block(self, prompt, expected_code, mock_post):
        """Test that only the last code block is executed."""
        mock_response = mock.Mock()
        mock_response.json.return_value = {"output": "test", "error": ""}
        mock_post.return_value = mock_response

        tool = tool_vllm.PythonCodeTool(api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>")

        tool(prompt)

        # Check that the API was called with the last code block
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertEqual(call_args[1]["json"]["code"], expected_code)


class TestToolUseLLMIntegration(unittest.TestCase):
    """Integration tests for ToolUseLLM class."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tool_use_llm_basic_generation(self):
        """Integration test for basic generation with ToolUseLLM."""
        from transformers import AutoTokenizer
        from vllm import SamplingParams

        # Create a simple tool for testing
        python_code_tool = tool_vllm.PythonCodeTool(
            api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>"
        )
        tools = {python_code_tool.end_str: python_code_tool}

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            stop=["</code>", "<endoftext>"],
            n=2,
            max_tokens=100,
            include_stop_str_in_output=True,
        )

        # Create the LLM instance
        model_name = "Qwen/Qwen2.5-7B"
        llm = tool_vllm.ToolUseLLM(
            tools=tools,
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=1000,
            max_tool_calls=3,
        )

        # Test prompts
        system_prompt = """Below is a conversation between an user and an assistant."""
        prompts = ["User: Hello, how are you?\nAssistant:"]
        prompts = [system_prompt + "\n\n" + p for p in prompts]

        # Tokenize and generate
        tok = AutoTokenizer.from_pretrained(model_name)
        prompt_token_ids = [tok.encode(p) for p in prompts]

        # Generate outputs
        outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

        # Basic assertions
        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0].outputs), 2)  # n=2

        # Check that output has expected attributes
        for output in outputs[0].outputs:
            self.assertTrue(hasattr(output, "mask"))
            self.assertTrue(hasattr(output, "num_calls"))
            self.assertTrue(hasattr(output, "timeout"))
            self.assertTrue(hasattr(output, "tool_error"))
            self.assertTrue(hasattr(output, "tool_output"))
            self.assertTrue(hasattr(output, "tool_runtime"))
            self.assertTrue(hasattr(output, "tool_called"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tool_use_llm_with_dataset(self):
        """Integration test using a real dataset."""
        from vllm import SamplingParams

        from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu

        # Create tools
        python_code_tool = tool_vllm.PythonCodeTool(
            api_endpoint="http://localhost:1212", start_str="<code>", end_str="</code>"
        )
        tools = {python_code_tool.end_str: python_code_tool}

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            stop=["</code>", "<endoftext>"],
            n=1,
            max_tokens=500,
            include_stop_str_in_output=True,
        )

        # Create the LLM instance
        model_name = "Qwen/Qwen2.5-7B"
        llm = tool_vllm.ToolUseLLM(
            tools=tools,
            model=model_name,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,
            max_model_len=5000,
            max_tool_calls=4,
        )

        # Load dataset
        tc = TokenizerConfig(
            tokenizer_name_or_path=model_name, chat_template_name="r1_simple_chat_postpend_think_tools7"
        )
        transform_fn_args = [{}, {"max_token_length": 8192, "max_prompt_token_length": 2048}]
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=["ai2-adapt-dev/rlvr_open_reasoner_math", "1.0"],
            dataset_mixer_list_splits=["train"],
            tc=tc,
            dataset_transform_fn=["rlvr_tokenize_v1", "rlvr_filter_v1"],
            transform_fn_args=transform_fn_args,
            dataset_cache_mode="local",
            hf_entity="allenai",
            dataset_local_cache_dir="/weka/oe-adapt-default/allennlp/deletable_open_instruct_dataset_cache",
        )

        # Generate outputs for a small subset
        outputs = llm.generate(prompt_token_ids=train_dataset["input_ids_prompt"][:2], sampling_params=sampling_params)

        # Verify outputs
        self.assertEqual(len(outputs), 2)

        # Check timeout and error rates
        timeouts = [o for output in outputs for o in output.outputs if o.timeout]
        errors = [o for output in outputs for o in output.outputs if len(o.tool_error) > 0]
        tool_called = [o for output in outputs for o in output.outputs if o.tool_called]

        # Basic sanity checks
        self.assertIsInstance(len(timeouts), int)
        self.assertIsInstance(len(errors), int)
        self.assertIsInstance(len(tool_called), int)


class TestToolUseLLMUnit(unittest.TestCase):
    """Unit tests for ToolUseLLM with mocked vLLM."""

    @mock.patch("open_instruct.tool_utils.tool_vllm.LLM.__init__")
    def test_tool_use_llm_initialization(self, mock_llm_init):
        """Test ToolUseLLM initialization."""
        mock_llm_init.return_value = None

        # Create mock tools
        mock_tool = mock.Mock()
        mock_tool.end_str = "</code>"
        tools = {"</code>": mock_tool}

        # Test with int max_tool_calls
        llm = tool_vllm.ToolUseLLM(tools=tools, max_tool_calls=5, model="test-model")

        self.assertEqual(llm.tools, tools)
        self.assertEqual(llm.max_tool_calls, {"</code>": 5})
        self.assertIsNotNone(llm.executor)
        self.assertEqual(llm.pending_tool_futures, {})

    @mock.patch("open_instruct.tool_utils.tool_vllm.LLM.__init__")
    def test_tool_use_llm_with_dict_max_calls(self, mock_llm_init):
        """Test ToolUseLLM initialization with dict max_tool_calls."""
        mock_llm_init.return_value = None

        # Create mock tools
        mock_tool1 = mock.Mock()
        mock_tool1.end_str = "</code>"
        mock_tool2 = mock.Mock()
        mock_tool2.end_str = "</tool>"

        tools = {"</code>": mock_tool1, "</tool>": mock_tool2}

        max_tool_calls = {"</code>": 3, "</tool>": 5}

        llm = tool_vllm.ToolUseLLM(tools=tools, max_tool_calls=max_tool_calls, model="test-model")

        self.assertEqual(llm.max_tool_calls, max_tool_calls)

    @mock.patch("open_instruct.tool_utils.tool_vllm.LLM.__init__")
    def test_validate_and_add_requests_overrides_n(self, mock_llm_init):
        """Test that _validate_and_add_requests overrides n=1."""
        # Mock the parent class init to avoid actual model loading
        mock_llm_init.return_value = None

        # Create the ToolUseLLM instance
        llm = tool_vllm.ToolUseLLM(tools={}, model="test-model")

        # Manually set up the required attributes that would normally be set by parent __init__
        mock_llm_engine = mock.Mock()
        llm.llm_engine = mock_llm_engine

        # Create sampling params with n > 1
        from vllm import SamplingParams

        sampling_params = SamplingParams(n=5, max_tokens=100)

        # Call _validate_and_add_requests
        prompts = ["test prompt"]
        llm._validate_and_add_requests(
            prompts=prompts, params=sampling_params, use_tqdm=False, lora_request=None, prompt_adapter_request=None
        )

        # Verify that the sampling params were modified to have n=1
        self.assertEqual(llm.single_n_sampling_params.n, 1)

        # Verify that add_request was called 5 times (original n value)
        self.assertEqual(mock_llm_engine.add_request.call_count, 5)


if __name__ == "__main__":
    unittest.main()
