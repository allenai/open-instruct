import asyncio
import subprocess
import time
import unittest

import ray

from open_instruct.tools.config import ToolArgs, build_tools_from_config
from open_instruct.tools.tools import (
    MaxCallsExceededTool,
    PythonCodeTool,
    PythonCodeToolConfig,
    S2SearchTool,
    S2SearchToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
)
from open_instruct.tools.utils import RetryConfig, ToolOutput, infer_tool_parameters


def run_async(coro):
    """Helper to run async code in tests."""
    return asyncio.run(coro)


class TestToolOutput(unittest.TestCase):
    def test_tool_output_creation(self):
        output = ToolOutput(output="test output", called=True, error="test error", timeout=False, runtime=1.5)
        self.assertEqual(output.output, "test output")
        self.assertTrue(output.called)
        self.assertEqual(output.error, "test error")
        self.assertFalse(output.timeout)
        self.assertEqual(output.runtime, 1.5)


class TestParameterInference(unittest.TestCase):
    """Test the parameter inference system."""

    def test_infer_tool_parameters_simple(self):
        """Test inferring parameters from a simple method."""
        from typing import Annotated

        from pydantic import Field

        def example_call(self, query: Annotated[str, Field(description="The search query")]) -> ToolOutput:
            """Example method."""
            pass

        schema = infer_tool_parameters(example_call)
        self.assertEqual(schema["type"], "object")
        self.assertIn("query", schema["properties"])
        self.assertEqual(schema["properties"]["query"]["type"], "string")
        self.assertEqual(schema["properties"]["query"]["description"], "The search query")
        self.assertEqual(schema["required"], ["query"])

    def test_infer_tool_parameters_with_defaults(self):
        """Test inferring parameters with default values."""
        from typing import Annotated

        from pydantic import Field

        def example_call(
            self,
            query: Annotated[str, Field(description="The search query")],
            num_results: Annotated[int, Field(description="Number of results")] = 10,
        ) -> ToolOutput:
            """Example method."""
            pass

        schema = infer_tool_parameters(example_call)
        self.assertEqual(schema["required"], ["query"])  # num_results has default, not required
        self.assertEqual(schema["properties"]["num_results"]["default"], 10)

    def test_infer_tool_parameters_multiple_types(self):
        """Test inferring parameters with various types."""

        def example_call(self, items: list[str], metadata: dict[str, int], enabled: bool = True) -> ToolOutput:
            """Example method."""
            pass

        schema = infer_tool_parameters(example_call)
        self.assertEqual(schema["properties"]["items"]["type"], "array")
        self.assertEqual(schema["properties"]["items"]["items"]["type"], "string")
        self.assertEqual(schema["properties"]["metadata"]["type"], "object")
        self.assertEqual(schema["properties"]["enabled"]["type"], "boolean")
        self.assertIn("items", schema["required"])
        self.assertIn("metadata", schema["required"])


class TestToolParameterSchemas(unittest.TestCase):
    """Test that tools have correct parameter schemas (inferred or explicit)."""

    def test_python_code_tool_parameters(self):
        """Test PythonCodeTool parameter schema."""
        tool = PythonCodeTool(api_endpoint="http://example.com")
        params = tool.tool_parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("code", params["properties"])
        self.assertEqual(params["properties"]["code"]["type"], "string")
        self.assertEqual(params["required"], ["code"])

    def test_serper_search_tool_parameters(self):
        """Test SerperSearchTool parameter schema."""
        tool = SerperSearchTool()
        params = tool.tool_parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("query", params["properties"])
        self.assertEqual(params["properties"]["query"]["type"], "string")
        self.assertEqual(params["required"], ["query"])

    def test_s2_search_tool_parameters(self):
        """Test S2SearchTool parameter schema."""
        tool = S2SearchTool()
        params = tool.tool_parameters

        self.assertEqual(params["type"], "object")
        self.assertIn("query", params["properties"])
        self.assertEqual(params["properties"]["query"]["type"], "string")
        self.assertEqual(params["required"], ["query"])

    def test_max_calls_exceeded_tool_parameters(self):
        """Test MaxCallsExceededTool has empty parameters (explicit schema)."""
        tool = MaxCallsExceededTool()
        params = tool.tool_parameters

        self.assertEqual(params["type"], "object")
        self.assertEqual(params["properties"], {})
        self.assertEqual(params["required"], [])


class TestMaxCallsExceededTool(unittest.TestCase):
    def test_max_calls_exceeded_output(self):
        tool = MaxCallsExceededTool()
        result = run_async(tool())

        self.assertIsInstance(result, ToolOutput)
        self.assertEqual(result.output, "Max tool calls exceeded.")
        self.assertFalse(result.called)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertEqual(result.runtime, 0)


class TestPythonCodeTool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Start the tool server for tests."""
        # Start the server in a subprocess
        cls.server_process = subprocess.Popen(
            ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "1212"],
            cwd="open_instruct/tools/code_server",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,  # Create new process group
        )
        # Wait for server to start
        time.sleep(3)
        cls.api_endpoint = "http://localhost:1212/execute"

    @classmethod
    def tearDownClass(cls):
        """Stop the tool server."""
        if not cls.server_process:
            return
        cls.server_process.terminate()
        try:
            cls.server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.server_process.kill()
            cls.server_process.wait()

    def setUp(self):
        self.api_endpoint = self.__class__.api_endpoint
        self.tool = PythonCodeTool(api_endpoint=self.api_endpoint)

    def test_initialization(self):
        self.assertEqual(self.tool.api_endpoint, self.api_endpoint)
        self.assertEqual(self.tool.tool_function_name, "python")

    def test_custom_override_name(self):
        """Test that override_name overrides the default tool_function_name."""
        tool = PythonCodeTool(api_endpoint=self.api_endpoint, override_name="code")
        self.assertEqual(tool.tool_function_name, "code")
        # Without override_name, should use default
        tool_default = PythonCodeTool(api_endpoint=self.api_endpoint)
        self.assertEqual(tool_default.tool_function_name, "python")

    def test_build_from_config(self):
        config = PythonCodeToolConfig(api_endpoint=self.api_endpoint, timeout=5)
        tool = config.build()
        self.assertEqual(tool.api_endpoint, self.api_endpoint)
        self.assertEqual(tool.timeout, 5)

    def test_successful_code_execution(self):
        result = run_async(self.tool(code='print("Hello, World!")'))

        self.assertTrue(result.called)
        self.assertIn("Hello, World!", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

    def test_code_execution_with_error(self):
        result = run_async(self.tool(code='print("unclosed string'))

        self.assertTrue(result.called)
        self.assertTrue("SyntaxError" in result.output or len(result.error) > 0)
        self.assertFalse(result.timeout)

    def test_timeout_handling(self):
        result = run_async(self.tool(code="import time\ntime.sleep(10)"))

        self.assertTrue(result.called)
        self.assertTrue(result.timeout or "Timeout" in result.output or "timeout" in result.error)
        self.assertLess(result.runtime, 10)  # Should timeout before 10 seconds

    def test_computation(self):
        result = run_async(self.tool(code='result = 5 * 7 + 3\nprint(f"The result is {result}")'))

        self.assertTrue(result.called)
        self.assertIn("The result is 38", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)


class TestOverrideNameOverride(unittest.TestCase):
    """Test the override_name override functionality for tools."""

    def test_serper_default_tag(self):
        """SerperSearchTool defaults to 'serper_search' tag."""
        tool = SerperSearchTool()
        self.assertEqual(tool.tool_function_name, "serper_search")

    def test_serper_custom_tag(self):
        """SerperSearchTool can use a custom tag."""
        tool = SerperSearchTool(override_name="google")
        self.assertEqual(tool.tool_function_name, "google")

    def test_s2_default_tag(self):
        """S2SearchTool defaults to 's2_search' tag."""
        tool = S2SearchTool()
        self.assertEqual(tool.tool_function_name, "s2_search")

    def test_s2_custom_tag_via_config(self):
        """S2SearchTool can use a custom tag via config."""
        config = S2SearchToolConfig(override_name="search")
        tool = config.build()
        self.assertEqual(tool.tool_function_name, "search")

    def test_serper_custom_tag_via_config(self):
        """SerperSearchTool can use a custom tag via config."""
        config = SerperSearchToolConfig(override_name="websearch")
        tool = config.build()
        self.assertEqual(tool.tool_function_name, "websearch")


class TestToolConfigOverrideName(unittest.TestCase):
    """Test override_name handling in ToolConfig and build_tools_from_config."""

    def test_single_tool_override_name_via_config(self):
        """Single tool with override_name set via tool_configs."""
        args = ToolArgs(tools=["s2_search"], tool_configs=['{"override_name": "search"}'])
        config = args.to_tool_config()
        tools, stop_strings = build_tools_from_config(config)
        # The tool should be keyed by the overridden tag name
        self.assertIn("search", tools)
        self.assertEqual(ray.get(tools["search"].get_tool_function_name.remote()), "search")

    def test_multiple_tools_with_override_names_via_config(self):
        """Multiple tools with override names set via tool_configs."""
        args = ToolArgs(
            tools=["s2_search", "serper_search"],
            tool_configs=['{"override_name": "academic_search"}', '{"override_name": "web_search"}'],
        )
        config = args.to_tool_config()
        tools, stop_strings = build_tools_from_config(config)
        self.assertIn("academic_search", tools)
        self.assertIn("web_search", tools)
        self.assertEqual(ray.get(tools["academic_search"].get_tool_function_name.remote()), "academic_search")
        self.assertEqual(ray.get(tools["web_search"].get_tool_function_name.remote()), "web_search")

    def test_no_override_names_uses_defaults(self):
        """When override_name is not provided, tools use their defaults."""
        args = ToolArgs(tools=["s2_search", "serper_search"], tool_configs=["{}", "{}"])
        config = args.to_tool_config()
        tools, stop_strings = build_tools_from_config(config)
        self.assertIn("s2_search", tools)
        self.assertIn("serper_search", tools)


class TestToolConfigs(unittest.TestCase):
    """Test the tool_configs list functionality."""

    def test_tool_configs_simple(self):
        """Test that tool_configs works for simple values."""
        args = ToolArgs(tools=["serper_search"], tool_configs=['{"num_results": 10}'])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config(0).num_results, 10)

    def test_tool_configs_multiple_fields(self):
        """Test that tool_configs works with multiple fields."""
        args = ToolArgs(tools=["s2_search"], tool_configs=['{"num_results": 20, "override_name": "papers"}'])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config(0).num_results, 20)
        self.assertEqual(config.get_tool_config(0).override_name, "papers")

    def test_tool_configs_multiple_tools(self):
        """Test tool_configs with multiple tools."""
        args = ToolArgs(
            tools=["serper_search", "s2_search"], tool_configs=['{"num_results": 5}', '{"num_results": 15}']
        )
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config(0).num_results, 5)
        self.assertEqual(config.get_tool_config(1).num_results, 15)

    def test_tool_configs_empty_dict_uses_defaults(self):
        """Test that empty dict {} uses default values."""
        args = ToolArgs(tools=["serper_search"], tool_configs=["{}"])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config(0).num_results, 5)  # default

    def test_tool_configs_invalid_json(self):
        """Test that invalid JSON raises an error."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tools=["serper_search"], tool_configs=['{"num_results": invalid}'])
        self.assertIn("Invalid JSON", str(context.exception))

    def test_tool_configs_not_dict(self):
        """Test that non-dict JSON raises an error."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tools=["serper_search"], tool_configs=['["a", "b"]'])
        self.assertIn("must be a JSON object", str(context.exception))

    def test_tool_configs_unknown_key(self):
        """Test that unknown keys in config raise an error."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tools=["serper_search"], tool_configs=['{"unknown_field": 123}'])
        self.assertIn("Unknown key", str(context.exception))

    def test_tool_configs_length_mismatch(self):
        """Test that mismatched lengths raise an error."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tools=["serper_search", "s2_search"], tool_configs=['{"num_results": 5}'])
        self.assertIn("same length", str(context.exception))

    def test_tool_configs_without_tools(self):
        """Test that tool_configs without tools raises an error."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tool_configs=['{"num_results": 5}'])
        self.assertIn("requires --tools", str(context.exception))


class TestToolActor(unittest.TestCase):
    """Test the ToolActor class."""

    @classmethod
    def setUpClass(cls):
        """Initialize Ray for actor tests."""
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray."""
        import ray

        if ray.is_initialized():
            ray.shutdown()

    def test_tool_actor_creation(self):
        """Test creating a ToolActor from config."""
        from open_instruct.tools.proxy import create_tool_actor_from_config

        config = SerperSearchToolConfig(num_results=5)

        # Create actor from config (config.build() called inside actor)
        actor = create_tool_actor_from_config(config=config)

        # Check we can get metadata via remote calls
        self.assertEqual(ray.get(actor.get_tool_function_name.remote()), "serper_search")

    def test_tool_actor_call(self):
        """Test that ToolActor correctly executes tool calls."""
        from open_instruct.tools.proxy import create_tool_actor_from_config

        config = SerperSearchToolConfig(num_results=5)
        actor = create_tool_actor_from_config(config=config)

        # Test that we can call the actor and get a result (async)
        result = ray.get(actor.call.remote(query="test query"))
        self.assertIsInstance(result, ToolOutput)
        # The result depends on the API, just check we got a response
        self.assertIsNotNone(result.output)


class TestRetryConfig(unittest.TestCase):
    """Test the RetryConfig dataclass."""

    def test_retry_config_defaults(self):
        """Test RetryConfig has sensible defaults."""
        config = RetryConfig()
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.backoff_factor, 0.5)
        self.assertIn(ConnectionError, config.retryable_exceptions)
        self.assertIn(TimeoutError, config.retryable_exceptions)

    def test_retry_config_custom(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(max_retries=5, backoff_factor=1.0, retryable_exceptions=(ValueError, KeyError))
        self.assertEqual(config.max_retries, 5)
        self.assertEqual(config.backoff_factor, 1.0)
        self.assertIn(ValueError, config.retryable_exceptions)
        self.assertIn(KeyError, config.retryable_exceptions)


if __name__ == "__main__":
    unittest.main()
