"""
A wrapper and registry for tools in rl-rag-mcp.
"""
from typing import List
import inspect
import asyncio
import os
import time
import httpx
import httpcore

try:
    from mcp_agents.tool_interface.mcp_tools import MassiveServeSearchTool, SemanticScholarSnippetSearchTool, SerperSearchTool, Crawl4AIBrowseTool
except ImportError as e:
    print(f"Failed to import mcp_agents. Please install from the source code:\n{e}")
    raise e

from open_instruct.search_rewards.utils.format_utils import generate_snippet_id
from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput

MCP_TOOL_REGISTRY = {
    "snippet_search": SemanticScholarSnippetSearchTool,
    "google_search": SerperSearchTool,
    "massive_serve": MassiveServeSearchTool,
    "browse_webpage": Crawl4AIBrowseTool
}

def truncate_at_second_last_stop(text: str, stops: list[str]) -> str:
    # dedup stop strings
    stops = list(set(stops))
    # Collect all stop occurrences (position, stopstring)
    positions = []
    for stop in stops:
        start = 0
        while True:
            idx = text.find(stop, start)
            if idx == -1:
                break
            positions.append((idx, stop))
            start = idx + len(stop)

    # If fewer than 2 stops, return unchanged
    if len(positions) < 2:
        return text

    # Sort by position in the string
    positions.sort(key=lambda x: x[0])

    # Take the second-last occurrence
    idx, stop = positions[-2]

    # Remove everything up to and including this occurrence
    return text[idx + len(stop):]


class MCPTool(Tool):
    """
    Unlike other tools, this guy handles *all mcp tools*. Why?
    because they share the same end string (</tool>). Hence, we need the parsers
    to work out how to route them. Ideally, this would be more tightly integrated into vllm,
    but for now, this is a bit cleaner.
    """
    def __init__(
        self,
        mcp_tool_names: List[str] | str,
        parser_name: str = "unified",
        transport_type: str | None = None,
        mcp_host: str | None = None,
        mcp_port: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        base_url: str | None = None,
        search_api_endpoint: str | None = None,
        start_str: str = "",
        end_str: str | None = None,
        *args,
        **kwargs,
    ):
        self.mcp_tools = []
        self.stop_strings = []
        # Allow selecting transport via arg or env; default to StreamableHttpTransport
        self.transport_type = transport_type or os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport")
        self.mcp_host = mcp_host or os.environ.get("MCP_TRANSPORT_HOST", "localhost")
        if self.base_url is not None:
            os.environ["MCP_TRANSPORT_HOST"] = self.mcp_host
        self.mcp_port = mcp_port or os.environ.get("MCP_TRANSPORT_PORT", 8000)
        if self.mcp_port is not None:
            os.environ["MCP_TRANSPORT_PORT"] = self.mcp_port
        # Prefer explicit base_url, fall back to search_api_endpoint for compatibility
        self.base_url = base_url or search_api_endpoint
        # Transient error retry settings
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        # Support comma-separated string for mcp_tool_names
        if isinstance(mcp_tool_names, str):
            mcp_tool_names = [n.strip() for n in mcp_tool_names.split(",") if n.strip()]
        for mcp_tool_name in mcp_tool_names:
            # filter kwargs so we only pass ones the constructor understands
            mcp_tool_cls = MCP_TOOL_REGISTRY[mcp_tool_name]
            sig = inspect.signature(mcp_tool_cls.__init__)
            valid_params = set(sig.parameters.keys())
            filtered_kwargs = {
                k: v for k, v in kwargs.items() if k in valid_params
            }

            # basically, we want to defer as much as possible to the mcp tool.
            # this 'tool' actually just passes everything down to the mcp tool.
            self.mcp_tools.append(mcp_tool_cls(
                name=mcp_tool_name,
                tool_parser=parser_name,
                transport_type=self.transport_type,
                **filtered_kwargs,
            ))
            # assign the stop strings from the parser itself.
            self.stop_strings += self.mcp_tools[-1].tool_parser.stop_sequences
        # MCP tool handles its own start and end strings.
        super().__init__(start_str=start_str, end_str=end_str or self.stop_strings[-1])

    def get_stop_strings(self) -> List[str]:
        return self.stop_strings

    def __call__(self, prompt: str) -> ToolOutput:
        # the one thing open-instruct needs to do: remove older tool calls.
        trunc_prompt = truncate_at_second_last_stop(prompt, self.stop_strings)
        # work out which mcp tool to call.
        document_tool_output = None
        error = None
        found_tool = False
        text_output = ""
        try:
            for mcp_tool in self.mcp_tools:
                if mcp_tool.tool_parser.has_calls(trunc_prompt, mcp_tool.name):
                    # Retry on transient stream/network errors
                    last_exc: Exception | None = None
                    for attempt in range(self.max_retries):
                        try:
                            document_tool_output = asyncio.run(mcp_tool(trunc_prompt))
                            break
                        except (httpcore.RemoteProtocolError, httpx.ReadError, ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                            last_exc = e
                            print(f"Error: {e}, retrying...")
                            if attempt + 1 >= self.max_retries:
                                raise
                            time.sleep(self.retry_backoff * (2 ** attempt))
                    # first format the output
                    text_output = mcp_tool._format_output(document_tool_output)
                    # wrap in the tags
                    text_output = mcp_tool.tool_parser.format_result(text_output, document_tool_output)
                    found_tool = True
                    break
        except Exception as e:
            error = str(e)
        if document_tool_output is None:
            if error is None and not found_tool:
                error = "No valid tool calls found."
                return ToolOutput(
                    output=error,
                    called=False,
                    error=error,
                    timeout=False,
                    runtime=0,
                    start_str="<snippet id=" + generate_snippet_id() + ">\n",
                    end_str="\n</snippet>",
                )
            elif error is not None:
                return ToolOutput(
                    output=error,
                    called=False,
                    error=error,
                    timeout=False,
                    runtime=0,
                    start_str="<snippet id=" + generate_snippet_id() + ">\n",
                    end_str="\n</snippet>",
                )
            else:
                print(f"Unknown error, no MCP response and no error found.")
                return ToolOutput(
                    output="Unknown error, no MCP response and no error found.",
                    called=False,
                    error="Unknown error, no MCP response and no error found.",
                    timeout=False,
                    runtime=0,
                    start_str="<snippet id=" + generate_snippet_id() + ">\n",
                    end_str="\n</snippet>",
                )

        if document_tool_output.error:
            print(f"Error from mcp tool: {document_tool_output.error}")
            print("Returning error output anyway.")
        # munge into format that open-instruct likes.
        print(f"Tool call prompt: {trunc_prompt}")
        print(f"Returning tool output: {text_output}")
        return ToolOutput(
            output=text_output,
            called=True,
            error=document_tool_output.error,
            timeout=document_tool_output.timeout,
            runtime=document_tool_output.runtime,
            start_str="\n",
            end_str="\n\n",
        )


if __name__ == "__main__":
    # example usage.
    from open_instruct.grpo_fast import launch_mcp_subprocess
    import time
    # need to launch mcp server first.
    launch_mcp_subprocess("uv run python -m rl-rag-mcp.mcp_agents.mcp_backend.main --transport http --port 8000 --host 0.0.0.0 --path /mcp", "./mcp_logs")
    # wait for it to launch.
    time.sleep(10)
    # then we can use the mcp tool.
    mcp_tool = MCPTool(["google_search"], number_documents_to_search=10, api_endpoint="http://localhost:8000/mcp")
    print(mcp_tool('<tool name="google_search">What is the capital of France?</tool>'))

