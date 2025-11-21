#!/usr/bin/env python
# /// script
# dependencies = [
#   "requests",
#   "openai",
# ]
# ///
"""Integration tests for vLLM server functionality.

Tests include:
- Reasoning content separation (TestVLLMReasoningSeparation)
  * Automatically starts Olmo-3-7B-Think with --reasoning-parser olmo3
  * Tests streaming and non-streaming reasoning separation

- Tool/function calling (TestVLLMToolUse)
  * Automatically starts Olmo-3-7B-Instruct with --enable-auto-tool-choice --tool-call-parser olmo3
  * Tests streaming/non-streaming tool calls and parallel tool calls

Server Management:
------------------

The vLLM server is automatically started and stopped by each test class.
No manual server setup is required.

Running Tests:
--------------

Example command (runs as a script with automatic dependency installation):
    uv run test_vllm_integration.py

Or with pytest for more control:
    uv run pytest test_vllm_integration.py -v

Run only reasoning tests:
    uv run pytest test_vllm_integration.py::TestVLLMReasoningSeparation -v

Run only tool use tests:
    uv run pytest test_vllm_integration.py::TestVLLMToolUse -v

Run a specific test:
    uv run pytest test_vllm_integration.py::TestVLLMToolUse::test_non_streaming_tool_call -v

Note: Each test class may take 5-10 minutes to start as it loads the model weights.
"""

import logging
import subprocess
import time
import unittest

import requests
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000/v1"
SERVER_STARTUP_TIMEOUT = 600


class TestVLLMReasoningSeparation(unittest.TestCase):
    """Test that vLLM properly separates reasoning content from regular content.

    Automatically starts/stops vLLM server with Olmo-3-7B-Think.
    """

    @classmethod
    def setUpClass(cls):
        """Start vLLM server and initialize OpenAI client."""
        logger.info("Starting vLLM server with Olmo-3-7B-Think...")
        cls.server_process = subprocess.Popen(
            ["uvx", "vllm", "serve", "allenai/Olmo-3-7B-Think", "--reasoning-parser", "olmo3"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info("Waiting for server to be ready...")
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                response = requests.get(f"{BASE_URL}/models", timeout=5)
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    break
            except requests.exceptions.RequestException:
                time.sleep(2)
        else:
            cls.server_process.kill()
            raise RuntimeError(f"Server did not start within {SERVER_STARTUP_TIMEOUT} seconds")

        cls.client = OpenAI(
            api_key="EMPTY",
            base_url=BASE_URL,
        )
        models = cls.client.models.list()
        cls.model = models.data[0].id
        logger.info(f"[Reasoning Tests] Testing model: {cls.model}")

    @classmethod
    def tearDownClass(cls):
        """Stop the vLLM server."""
        logger.info("Stopping vLLM server...")
        cls.server_process.terminate()
        try:
            cls.server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully, killing...")
            cls.server_process.kill()
        logger.info("Server stopped.")

    def test_streaming_reasoning_separation(self):
        """Test that streaming mode properly separates reasoning_content."""
        messages = [
            {"role": "user", "content": "9.11 and 9.8, which is greater?"}
        ]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=True,
        )

        chunks_with_content = 0
        chunks_with_reasoning = 0
        total_chunks = 0

        for chunk in stream:
            total_chunks += 1

            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                if hasattr(delta, 'content') and delta.content is not None:
                    chunks_with_content += 1

                if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                    chunks_with_reasoning += 1

        logger.info(f"Streaming - Total chunks: {total_chunks}, "
                   f"content: {chunks_with_content}, "
                   f"reasoning_content: {chunks_with_reasoning}")

        self.assertGreater(chunks_with_reasoning, 0,
                          "Expected at least one chunk with reasoning_content in streaming mode")
        self.assertGreater(total_chunks, 0,
                          "Expected to receive chunks from the model")

    def test_non_streaming_reasoning_separation(self):
        """Test that non-streaming mode properly separates reasoning field."""
        messages = [
            {"role": "user", "content": "9.11 and 9.8, which is greater?"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=500,
            temperature=0.7,
            stream=False,
        )

        self.assertTrue(response.choices, "Expected at least one choice in response")
        self.assertGreater(len(response.choices), 0, "Expected non-empty choices list")

        choice = response.choices[0]
        message = choice.message

        reasoning = getattr(message, 'reasoning', None)
        content = getattr(message, 'content', None)

        logger.info(f"Non-streaming - Has reasoning: {reasoning is not None}, "
                   f"Has content: {content is not None}")

        if reasoning is not None:
            logger.info(f"Reasoning length: {len(reasoning)}")
        if content is not None:
            logger.info(f"Content length: {len(content)}")

        self.assertIsNotNone(reasoning,
                           "Expected 'reasoning' field to be present and non-null in non-streaming response")
        self.assertIsNotNone(content,
                           "Expected 'content' field to be present and non-null in non-streaming response")


class TestVLLMToolUse(unittest.TestCase):
    """Test that vLLM properly handles tool/function calling.

    Automatically starts/stops vLLM server with Olmo-3-7B-Instruct.
    """

    @classmethod
    def setUpClass(cls):
        """Start vLLM server and initialize OpenAI client with tools."""
        logger.info("Starting vLLM server with Olmo-3-7B-Instruct...")
        cls.server_process = subprocess.Popen(
            ["uvx", "vllm", "serve", "allenai/Olmo-3-7B-Instruct",
             "--enable-auto-tool-choice", "--tool-call-parser", "olmo3"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        logger.info("Waiting for server to be ready...")
        start_time = time.time()
        while time.time() - start_time < SERVER_STARTUP_TIMEOUT:
            try:
                response = requests.get(f"{BASE_URL}/models", timeout=5)
                if response.status_code == 200:
                    logger.info("Server is ready!")
                    break
            except requests.exceptions.RequestException:
                time.sleep(2)
        else:
            cls.server_process.kill()
            raise RuntimeError(f"Server did not start within {SERVER_STARTUP_TIMEOUT} seconds")

        cls.client = OpenAI(
            api_key="EMPTY",
            base_url=BASE_URL,
        )
        models = cls.client.models.list()
        cls.model = models.data[0].id
        logger.info(f"[Tool Use Tests] Testing model: {cls.model}")

        cls.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                                "description": "The temperature unit to use",
                            },
                        },
                        "required": ["location"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Perform a mathematical calculation",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "The mathematical expression to evaluate",
                            },
                        },
                        "required": ["expression"],
                    },
                },
            },
        ]

    @classmethod
    def tearDownClass(cls):
        """Stop the vLLM server."""
        logger.info("Stopping vLLM server...")
        cls.server_process.terminate()
        try:
            cls.server_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Server did not stop gracefully, killing...")
            cls.server_process.kill()
        logger.info("Server stopped.")

    def test_non_streaming_tool_call(self):
        """Test that non-streaming mode properly returns tool calls."""
        messages = [
            {"role": "user", "content": "What's the weather like in San Francisco?"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            max_tokens=500,
            temperature=0.7,
            stream=False,
        )

        self.assertTrue(response.choices, "Expected at least one choice in response")
        self.assertGreater(len(response.choices), 0, "Expected non-empty choices list")

        choice = response.choices[0]
        message = choice.message

        tool_calls = getattr(message, 'tool_calls', None)

        logger.info(f"Non-streaming - Has tool_calls: {tool_calls is not None}")
        if tool_calls:
            logger.info(f"Number of tool calls: {len(tool_calls)}")
            for idx, tool_call in enumerate(tool_calls):
                logger.info(f"Tool call {idx}: {tool_call.function.name}")
                logger.info(f"  Arguments: {tool_call.function.arguments}")

        self.assertIsNotNone(tool_calls,
                           "Expected 'tool_calls' field to be present in response")
        self.assertGreater(len(tool_calls), 0,
                          "Expected at least one tool call for weather query")

        first_tool_call = tool_calls[0]
        self.assertEqual(first_tool_call.function.name, "get_weather",
                        "Expected model to call 'get_weather' function")
        self.assertIsNotNone(first_tool_call.function.arguments,
                           "Expected tool call to have arguments")

    def test_streaming_tool_call(self):
        """Test that streaming mode properly returns tool calls."""
        messages = [
            {"role": "user", "content": "Calculate 25 * 4 + 10"}
        ]

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            max_tokens=500,
            temperature=0.7,
            stream=True,
        )

        chunks_with_tool_calls = 0
        total_chunks = 0
        accumulated_tool_calls = []

        for chunk in stream:
            total_chunks += 1

            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta

                if hasattr(delta, 'tool_calls') and delta.tool_calls is not None:
                    chunks_with_tool_calls += 1
                    for tool_call in delta.tool_calls:
                        if tool_call.index is not None:
                            while len(accumulated_tool_calls) <= tool_call.index:
                                accumulated_tool_calls.append({
                                    'id': None,
                                    'function': {'name': '', 'arguments': ''}
                                })

                            if tool_call.id:
                                accumulated_tool_calls[tool_call.index]['id'] = tool_call.id
                            if tool_call.function:
                                if tool_call.function.name:
                                    accumulated_tool_calls[tool_call.index]['function']['name'] = tool_call.function.name
                                if tool_call.function.arguments:
                                    accumulated_tool_calls[tool_call.index]['function']['arguments'] += tool_call.function.arguments

        logger.info(f"Streaming - Total chunks: {total_chunks}, "
                   f"chunks with tool_calls: {chunks_with_tool_calls}")

        if accumulated_tool_calls:
            logger.info(f"Accumulated tool calls: {len(accumulated_tool_calls)}")
            for idx, tool_call in enumerate(accumulated_tool_calls):
                logger.info(f"Tool call {idx}: {tool_call['function']['name']}")
                logger.info(f"  Arguments: {tool_call['function']['arguments']}")

        self.assertGreater(chunks_with_tool_calls, 0,
                          "Expected at least one chunk with tool_calls in streaming mode")
        self.assertGreater(len(accumulated_tool_calls), 0,
                          "Expected at least one tool call to be accumulated")

        first_tool_call = accumulated_tool_calls[0]
        self.assertEqual(first_tool_call['function']['name'], "calculate",
                        "Expected model to call 'calculate' function")

    def test_parallel_tool_calls(self):
        """Test that the model can make multiple parallel tool calls."""
        messages = [
            {"role": "user", "content": "What's the weather in New York and calculate 15 + 27?"}
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.tools,
            max_tokens=500,
            temperature=0.7,
            stream=False,
        )

        choice = response.choices[0]
        message = choice.message
        tool_calls = getattr(message, 'tool_calls', None)

        logger.info(f"Parallel tool calls - Number of tool calls: {len(tool_calls) if tool_calls else 0}")

        if tool_calls:
            function_names = [tc.function.name for tc in tool_calls]
            logger.info(f"Functions called: {function_names}")

        self.assertIsNotNone(tool_calls, "Expected tool_calls to be present")
        self.assertGreaterEqual(len(tool_calls), 1,
                               "Expected at least one tool call (ideally two for parallel calls)")


if __name__ == "__main__":
    unittest.main()
