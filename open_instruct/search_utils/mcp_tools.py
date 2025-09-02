"""
A wrapper and registry for tools in rl-rag-mcp.
"""
import inspect
import asyncio

from mcp_agents.tool_interface.mcp_tools import MassiveServeSearchTool, SemanticScholarSnippetSearchTool, SerperSearchTool, SerperBrowseTool

from open_instruct.search_rewards.format_utils import generate_snippet_id
from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput

MCP_TOOL_REGISTRY = {
    "snippet_search": SemanticScholarSnippetSearchTool,
    "google_search": SerperSearchTool,
    "massive_serve": MassiveServeSearchTool,
    "browse_webpage": SerperBrowseTool
}

def truncate_at_second_last_stop(text: str, stops: list[str]) -> str:
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
    def __init__(self, mcp_tool_name: str, parser_name: str = "unified", *args, **kwargs):
        # filter kwargs so we only pass ones the constructor understands
        mcp_tool_cls = MCP_TOOL_REGISTRY[mcp_tool_name]
        sig = inspect.signature(mcp_tool_cls.__init__)
        valid_params = set(sig.parameters.keys())
        filtered_kwargs = {
            k: v for k, v in kwargs.items() if k in valid_params
        }

        # basically, we want to defer as much as possible to the mcp tool.
        # this 'tool' actually just passes everything down to the mcp tool.
        self.mcp_tool = mcp_tool_cls(
            tool_parser=parser_name,
            transport_type="StreamableHttpTransport",  # for now, we only support streamable http transport.
            **filtered_kwargs,
        )
        # assign the stop strings from the parser itself.
        self.stop_strings = self.mcp_tool.tool_parser.stop_sequences
        # MCP tool handles its own start and end strings.
        super().__init__(start_str="", end_str=self.stop_strings[-1])

    def __call__(self, prompt: str) -> ToolOutput:
        # the one thing open-instruct needs to do: remove older tool calls.
        trunc_prompt = truncate_at_second_last_stop(prompt, self.stop_strings)
        print(f"trunc_prompt: {trunc_prompt}")
        # then we just directly call the tool!
        document_tool_output = asyncio.run(self.mcp_tool(trunc_prompt))
        # mcp tool return is Optional[DocumentToolOutput]
        if document_tool_output is None:
            return ToolOutput(
                output="Error calling tool.",
                called=False,
                error="Error calling tool.",
                timeout=False,
                runtime=0,
                start_str="<snippet id=" + generate_snippet_id() + ">\n",
                end_str="\n</snippet>",
            )

        # munge into format that open-instruct likes.
        return ToolOutput(
            output=document_tool_output.output,
            called=True,
            error=document_tool_output.error,
            timeout=document_tool_output.timeout,
            runtime=document_tool_output.runtime,
            start_str="<snippet id=" + generate_snippet_id() + ">\n",
            end_str="\n</snippet>",
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
    mcp_tool = MCPTool("serper", number_documents_to_search=10, api_endpoint="http://localhost:8000/mcp")
    print(mcp_tool('<tool name="SerperSearchTool">What is the capital of France?</tool>'))

