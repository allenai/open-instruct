"""Tests for tool_vllm module."""

import unittest
from unittest import mock

import torch
from transformers import AutoTokenizer
from vllm import SamplingParams

from open_instruct.dataset_transformation import TokenizerConfig, get_cached_dataset_tulu
from open_instruct.tool_utils import tool_vllm


class TestToolUseLLMIntegration(unittest.TestCase):
    """Integration tests for ToolUseLLM class."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_tool_use_llm_basic_generation(self):
        """Integration test for basic generation with ToolUseLLM."""
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


def create_mock_request_output(request_id, prompt_token_ids, output_tokens, output_text):
    """Helper to create mock RequestOutput with proper structure."""
    mock_output = mock.Mock()
    mock_output.request_id = request_id
    mock_output.prompt_token_ids = prompt_token_ids
    mock_output.outputs = []

    # Create mock completion output
    completion = mock.Mock()
    completion.token_ids = output_tokens
    completion.text = output_text
    # Add the custom attributes that ToolUseLLM adds
    completion.mask = []
    completion.num_calls = 0
    completion.timeout = False
    completion.tool_error = ""
    completion.tool_output = ""
    completion.tool_runtime = 0.0
    completion.tool_called = False

    mock_output.outputs.append(completion)
    return mock_output


class TestToolUseLLMWithMockedVLLM(unittest.TestCase):
    """Integration tests with mocked vLLM - same as TestToolUseLLMIntegration but runs without GPU."""

    @mock.patch("vllm.LLM.generate")
    @mock.patch("vllm.LLM.__init__")
    def test_tool_use_llm_basic_generation(self, mock_init, mock_generate):
        """Integration test for basic generation with mocked vLLM."""
        # Mock init to do nothing
        mock_init.return_value = None

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

        # Tokenize (mock tokenization)
        tok = AutoTokenizer.from_pretrained(model_name)
        prompt_token_ids = [tok.encode(p) for p in prompts]

        # Create mock outputs - one output with 2 completions (n=2)
        mock_output = create_mock_request_output(
            request_id="0-0",
            prompt_token_ids=prompt_token_ids[0],
            output_tokens=[1, 2, 3, 4, 5],  # Mock token IDs
            output_text="I'm doing well, thank you!",
        )
        # Add second completion for n=2
        completion2 = mock.Mock()
        completion2.token_ids = [1, 2, 3, 6, 7]
        completion2.text = "Hello! I'm happy to help."
        completion2.mask = []
        completion2.num_calls = 0
        completion2.timeout = False
        completion2.tool_error = ""
        completion2.tool_output = ""
        completion2.tool_runtime = 0.0
        completion2.tool_called = False
        mock_output.outputs.append(completion2)

        mock_generate.return_value = [mock_output]

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

    @mock.patch("vllm.LLM.generate")
    @mock.patch("vllm.LLM.__init__")
    def test_tool_use_llm_with_dataset(self, mock_init, mock_generate):
        """Integration test using a dataset with mocked vLLM."""
        # Mock init to do nothing
        mock_init.return_value = None

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

        # Use mock dataset instead of loading real one to avoid directory issues
        # Create a mock dataset with the required structure
        train_dataset = {
            "input_ids_prompt": [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            "ground_truth": ["answer 1", "answer 2"],
        }

        # Create mock outputs for 2 prompts
        mock_outputs = []
        for i in range(2):
            mock_output = create_mock_request_output(
                request_id=f"{i}-0",
                prompt_token_ids=train_dataset["input_ids_prompt"][i]
                if i < len(train_dataset["input_ids_prompt"])
                else [1, 2, 3],
                output_tokens=[10 + i, 20 + i, 30 + i],
                output_text=f"Mock response {i}",
            )
            mock_outputs.append(mock_output)

        mock_generate.return_value = mock_outputs

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


if __name__ == "__main__":
    unittest.main()
