import subprocess
import time
import unittest

from open_instruct.tools.config import ToolArgs, ToolConfig, build_tools_from_config
from open_instruct.tools.tools import (
    MaxCallsExceededTool,
    PythonCodeTool,
    PythonCodeToolConfig,
    S2SearchTool,
    S2SearchToolConfig,
    SerperSearchTool,
    SerperSearchToolConfig,
)
from open_instruct.tools.utils import ToolOutput


class TestToolOutput(unittest.TestCase):
    def test_tool_output_creation(self):
        output = ToolOutput(output="test output", called=True, error="test error", timeout=False, runtime=1.5)
        self.assertEqual(output.output, "test output")
        self.assertTrue(output.called)
        self.assertEqual(output.error, "test error")
        self.assertFalse(output.timeout)
        self.assertEqual(output.runtime, 1.5)


class TestMaxCallsExceededTool(unittest.TestCase):
    def test_max_calls_exceeded_output(self):
        tool = MaxCallsExceededTool()
        result = tool()

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
        config = PythonCodeToolConfig(api_endpoint=self.api_endpoint, timeout_seconds=5)
        tool = config.build()
        self.assertEqual(tool.api_endpoint, self.api_endpoint)
        self.assertEqual(tool.timeout_seconds, 5)

    def test_successful_code_execution(self):
        result = self.tool(text='print("Hello, World!")')

        self.assertTrue(result.called)
        self.assertIn("Hello, World!", result.output)
        self.assertEqual(result.error, "")
        self.assertFalse(result.timeout)
        self.assertGreater(result.runtime, 0)

    def test_code_execution_with_error(self):
        result = self.tool(text='print("unclosed string')

        self.assertTrue(result.called)
        self.assertTrue("SyntaxError" in result.output or len(result.error) > 0)
        self.assertFalse(result.timeout)

    def test_timeout_handling(self):
        result = self.tool(text="import time\ntime.sleep(10)")

        self.assertTrue(result.called)
        self.assertTrue(result.timeout or "Timeout" in result.output or "timeout" in result.error)
        self.assertLess(result.runtime, 10)  # Should timeout before 10 seconds

    def test_computation(self):
        result = self.tool(text='result = 5 * 7 + 3\nprint(f"The result is {result}")')

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

    def test_single_tool_override_name_override(self):
        """Single tool with override_name override."""
        config = ToolConfig(tools=["s2_search"], tool_override_names=["search"])
        tools, stop_strings = build_tools_from_config(config)
        # The tool should be keyed by the overridden tag name
        self.assertIn("search", tools)
        self.assertEqual(tools["search"].tool_function_name, "search")

    def test_multiple_tools_with_override_names(self):
        """Multiple tools with corresponding override names."""
        config = ToolConfig(
            tools=["s2_search", "serper_search"], tool_override_names=["academic_search", "web_search"]
        )
        tools, stop_strings = build_tools_from_config(config)
        self.assertIn("academic_search", tools)
        self.assertIn("web_search", tools)
        self.assertEqual(tools["academic_search"].tool_function_name, "academic_search")
        self.assertEqual(tools["web_search"].tool_function_name, "web_search")

    def test_tool_args_to_config_override_names(self):
        """ToolArgs.to_tool_config() passes through override_names list."""
        args = ToolArgs(tools=["serper_search", "s2_search"], tool_override_names=["search", "papers"])
        config = args.to_tool_config()
        self.assertEqual(config.tool_override_names, ["search", "papers"])

    def test_tool_args_validation_mismatched_lengths(self):
        """ToolArgs raises error when tool_override_names length doesn't match tools."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tools=["serper_search", "s2_search"], tool_override_names=["search"])
        self.assertIn("same length", str(context.exception))

    def test_tool_args_validation_override_names_without_tools(self):
        """ToolArgs raises error when tool_override_names provided without tools."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tool_override_names=["search"])
        self.assertIn("requires --tools", str(context.exception))

    def test_no_override_names_uses_defaults(self):
        """When tool_override_names is not provided, tools use their defaults."""
        config = ToolConfig(tools=["s2_search", "serper_search"])
        tools, stop_strings = build_tools_from_config(config)
        self.assertIn("s2_search", tools)
        self.assertIn("serper_search", tools)  # serper default is "serper_search"


class TestToolConfigs(unittest.TestCase):
    """Test the tool_configs list functionality."""

    def test_tool_configs_simple(self):
        """Test that tool_configs works for simple values."""
        args = ToolArgs(tools=["serper_search"], tool_configs=['{"num_results": 10}'])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config("serper_search").num_results, 10)

    def test_tool_configs_multiple_fields(self):
        """Test that tool_configs works with multiple fields."""
        args = ToolArgs(tools=["s2_search"], tool_configs=['{"num_results": 20, "override_name": "papers"}'])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config("s2_search").num_results, 20)
        self.assertEqual(config.get_tool_config("s2_search").override_name, "papers")

    def test_tool_configs_multiple_tools(self):
        """Test tool_configs with multiple tools."""
        args = ToolArgs(
            tools=["serper_search", "s2_search"], tool_configs=['{"num_results": 5}', '{"num_results": 15}']
        )
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config("serper_search").num_results, 5)
        self.assertEqual(config.get_tool_config("s2_search").num_results, 15)

    def test_tool_configs_empty_dict_uses_defaults(self):
        """Test that empty dict {} uses default values."""
        args = ToolArgs(tools=["serper_search"], tool_configs=["{}"])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config("serper_search").num_results, 5)  # default

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

    def test_tool_configs_mcp(self):
        """Test tool_configs for MCP tool."""
        args = ToolArgs(tools=["mcp"], tool_configs=['{"tool_names": "snippet_search,google_search", "timeout": 120}'])
        config = args.to_tool_config()
        self.assertEqual(config.get_tool_config("mcp").tool_names, "snippet_search,google_search")
        self.assertEqual(config.get_tool_config("mcp").timeout, 120)


class TestToolProxy(unittest.TestCase):
    """Test the ToolProxy and ToolActor classes."""

    @classmethod
    def setUpClass(cls):
        """Initialize Ray for proxy tests."""
        import ray

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True, num_cpus=2)

    @classmethod
    def tearDownClass(cls):
        """Shutdown Ray."""
        import ray

        if ray.is_initialized():
            ray.shutdown()

    def test_tool_proxy_creation(self):
        """Test creating a ToolProxy from config using the two-step pattern."""
        from open_instruct.tools.proxy import ToolProxy, create_tool_actor_from_config

        config = SerperSearchToolConfig(num_results=5)

        # Step 1: Create actor from config (config.build() called inside actor)
        actor = create_tool_actor_from_config(config=config)

        # Step 2: Wrap with proxy
        proxy = ToolProxy.from_actor(actor)

        self.assertIsInstance(proxy, ToolProxy)
        self.assertEqual(proxy.tool_function_name, "serper_search")

    def test_tool_proxy_call(self):
        """Test that ToolProxy correctly forwards calls to the ToolActor."""
        from open_instruct.tools.proxy import ToolProxy, create_tool_actor_from_config

        config = SerperSearchToolConfig(num_results=5)
        actor = create_tool_actor_from_config(config=config)
        proxy = ToolProxy.from_actor(actor)

        # Test that we can call the proxy and get a result
        result = proxy(text="test query")
        self.assertIsInstance(result, ToolOutput)
        # The result depends on the API, just check we got a response
        self.assertIsNotNone(result.output)


if __name__ == "__main__":
    unittest.main()
