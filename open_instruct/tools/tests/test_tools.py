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

    def test_custom_tag_name(self):
        """Test that tag_name overrides the default tool_function_name."""
        tool = PythonCodeTool(api_endpoint=self.api_endpoint, tag_name="code")
        self.assertEqual(tool.tool_function_name, "code")
        # Without tag_name, should use default
        tool_default = PythonCodeTool(api_endpoint=self.api_endpoint)
        self.assertEqual(tool_default.tool_function_name, "python")

    def test_from_config(self):
        config = PythonCodeToolConfig(api_endpoint=self.api_endpoint, timeout_seconds=5)
        tool = PythonCodeTool.from_config(config)
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


class TestTagNameOverride(unittest.TestCase):
    """Test the tag_name override functionality for tools."""

    def test_serper_default_tag(self):
        """SerperSearchTool defaults to 'serper_search' tag."""
        tool = SerperSearchTool()
        self.assertEqual(tool.tool_function_name, "serper_search")

    def test_serper_custom_tag(self):
        """SerperSearchTool can use a custom tag."""
        tool = SerperSearchTool(tag_name="google")
        self.assertEqual(tool.tool_function_name, "google")

    def test_s2_default_tag(self):
        """S2SearchTool defaults to 's2_search' tag."""
        tool = S2SearchTool()
        self.assertEqual(tool.tool_function_name, "s2_search")

    def test_s2_custom_tag_via_config(self):
        """S2SearchTool can use a custom tag via config."""
        config = S2SearchToolConfig(tag_name="search")
        tool = S2SearchTool.from_config(config)
        self.assertEqual(tool.tool_function_name, "search")

    def test_serper_custom_tag_via_config(self):
        """SerperSearchTool can use a custom tag via config."""
        config = SerperSearchToolConfig(tag_name="websearch")
        tool = SerperSearchTool.from_config(config)
        self.assertEqual(tool.tool_function_name, "websearch")


class TestToolConfigTagName(unittest.TestCase):
    """Test tag_name handling in ToolConfig and build_tools_from_config."""

    def test_single_tool_tag_name_override(self):
        """Single tool with tag_name override."""
        config = ToolConfig(tools=["s2_search"], tool_tag_names=["search"], s2_search=S2SearchToolConfig())
        tools, stop_strings = build_tools_from_config(config)
        # The tool should be keyed by the overridden tag name
        self.assertIn("search", tools)
        self.assertEqual(tools["search"].tool_function_name, "search")

    def test_multiple_tools_with_tag_names(self):
        """Multiple tools with corresponding tag names."""
        config = ToolConfig(tools=["s2_search", "serper_search"], tool_tag_names=["academic_search", "web_search"])
        tools, stop_strings = build_tools_from_config(config)
        self.assertIn("academic_search", tools)
        self.assertIn("web_search", tools)
        self.assertEqual(tools["academic_search"].tool_function_name, "academic_search")
        self.assertEqual(tools["web_search"].tool_function_name, "web_search")

    def test_tool_args_to_config_tag_names(self):
        """ToolArgs.to_tool_config() passes through tag_names list."""
        args = ToolArgs(tools=["serper_search", "s2_search"], tool_tag_names=["search", "papers"])
        config = args.to_tool_config()
        self.assertEqual(config.tool_tag_names, ["search", "papers"])

    def test_tool_args_validation_mismatched_lengths(self):
        """ToolArgs raises error when tool_tag_names length doesn't match tools."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tools=["serper_search", "s2_search"], tool_tag_names=["search"])
        self.assertIn("same length", str(context.exception))

    def test_tool_args_validation_tag_names_without_tools(self):
        """ToolArgs raises error when tool_tag_names provided without tools."""
        with self.assertRaises(ValueError) as context:
            ToolArgs(tool_tag_names=["search"])
        self.assertIn("requires --tools", str(context.exception))

    def test_no_tag_names_uses_defaults(self):
        """When tool_tag_names is not provided, tools use their defaults."""
        config = ToolConfig(tools=["s2_search", "serper_search"])
        tools, stop_strings = build_tools_from_config(config)
        self.assertIn("s2_search", tools)
        self.assertIn("serper_search", tools)  # serper default is "serper_search"


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

        # Step 1: Create actor from config
        actor = create_tool_actor_from_config(class_path="open_instruct.tools.tools:SerperSearchTool", config=config)

        # Step 2: Wrap with proxy
        proxy = ToolProxy.from_actor(actor)

        self.assertIsInstance(proxy, ToolProxy)
        self.assertEqual(proxy.tool_function_name, "serper_search")

    def test_tool_proxy_call(self):
        """Test that ToolProxy correctly forwards calls to the ToolActor."""
        from open_instruct.tools.proxy import ToolProxy, create_tool_actor_from_config

        config = SerperSearchToolConfig(num_results=5)
        actor = create_tool_actor_from_config(class_path="open_instruct.tools.tools:SerperSearchTool", config=config)
        proxy = ToolProxy.from_actor(actor)

        # Test that we can call the proxy and get a result
        result = proxy(text="test query")
        self.assertIsInstance(result, ToolOutput)
        # The result depends on the API, just check we got a response
        self.assertIsNotNone(result.output)


if __name__ == "__main__":
    unittest.main()
