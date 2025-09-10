"""
A wrapper and registry for tools in rl-rag-mcp.
"""
from typing import List
import inspect
import asyncio
import threading
import time
import random
import logging

import httpx
import httpcore

try:
    from mcp_agents.tool_interface.mcp_tools import (
        MassiveServeSearchTool,
        SemanticScholarSnippetSearchTool,
        SerperSearchTool,
        Crawl4AIBrowseTool,
    )
except ImportError as e:
    print(f"Failed to import mcp_agents. Please install from the source code:\n{e}")
    raise e

from open_instruct.search_rewards.utils.format_utils import generate_snippet_id
from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput

log = logging.getLogger(__name__)

MCP_TOOL_REGISTRY = {
    "snippet_search": SemanticScholarSnippetSearchTool,
    "google_search": SerperSearchTool,
    "massive_serve": MassiveServeSearchTool,
    "browse_webpage": Crawl4AIBrowseTool,
}

TRANSIENT_NET_ERRORS = (
    httpx.RemoteProtocolError,   # <- the one your client logs show
    httpx.ReadError,
    httpx.ConnectError,
    httpx.WriteError,
    httpcore.RemoteProtocolError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)

def truncate_at_second_last_stop(text: str, stops: list[str]) -> str:
    # de-dup stops
    stops = list(set(stops))
    positions: list[tuple[int, str]] = []
    for stop in stops:
        start = 0
        while True:
            idx = text.find(stop, start)
            if idx == -1:
                break
            positions.append((idx, stop))
            start = idx + len(stop)
    if len(positions) < 2:
        return text
    positions.sort(key=lambda x: x[0])
    idx, stop = positions[-2]
    # Keep only the last tool-call segment (after the second-last stop).
    return text[idx + len(stop):]


def _run_async_safely(coro):
    """
    Run an async coroutine from possibly async contexts without tripping on
    'asyncio.run() cannot be called from a running event loop'.

    Strategy:
    - If no loop is running in this thread: asyncio.run(coro)
    - If a loop *is* running: spin a short-lived thread and run asyncio.run there.
      (Yes, this is fine. No, you won't like it aesthetically. It works.)
    """
    try:
        asyncio.get_running_loop()
        loop_running_here = True
    except RuntimeError:
        loop_running_here = False

    if not loop_running_here:
        return asyncio.run(coro)

    result_holder = {}
    exc_holder = {}

    def runner():
        try:
            result_holder["value"] = asyncio.run(coro)
        except Exception as e:  # noqa: BLE001
            exc_holder["error"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "error" in exc_holder:
        raise exc_holder["error"]
    return result_holder.get("value")


class MCPTool(Tool):
    """
    Routes calls to multiple MCP tools that share the same end tag.
    """
    def __init__(
        self,
        mcp_tool_names: List[str],
        parser_name: str = "unified",
        transport_type: str | None = "StreamableHttpTransport",
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        *args,
        **kwargs,
    ):
        self.mcp_tools = []
        all_stops: list[str] = []

        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        for mcp_tool_name in mcp_tool_names:
            mcp_tool_cls = MCP_TOOL_REGISTRY[mcp_tool_name]
            sig = inspect.signature(mcp_tool_cls.__init__)
            valid_params = set(sig.parameters.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            tool = mcp_tool_cls(
                name=mcp_tool_name,
                tool_parser=parser_name,
                transport_type=transport_type or "StreamableHttpTransport",
                **filtered_kwargs,
            )
            self.mcp_tools.append(tool)
            all_stops.extend(tool.tool_parser.stop_sequences)

        # Dedup and sort stop strings by length desc so longer sentinels win.
        dedup = sorted(set(all_stops), key=lambda s: (-len(s), s))
        self.stop_strings = dedup

        # Use a conservative end_str that matches your parser’s end token explicitly.
        # Fall back to the first (longest) stop.
        end_token = None
        for s in dedup:
            if s.strip().endswith("</tool>") or s.strip().endswith("</call_tool>"):
                end_token = s
                break
        super().__init__(start_str="", end_str=end_token or dedup[0])

    def get_stop_strings(self) -> List[str]:
        return self.stop_strings

    def __call__(self, prompt: str) -> ToolOutput:
        trunc_prompt = truncate_at_second_last_stop(prompt, self.stop_strings)

        document_tool_output = None
        found_tool = False
        text_output = ""
        error = None

        try:
            for mcp_tool in self.mcp_tools:
                if not mcp_tool.tool_parser.has_calls(trunc_prompt, mcp_tool.name):
                    continue

                # Transient error retries with jittered exponential backoff
                last_exc = None
                for attempt in range(self.max_retries):
                    try:
                        document_tool_output = _run_async_safely(mcp_tool(trunc_prompt))
                        break
                    except TRANSIENT_NET_ERRORS as e:
                        last_exc = e
                        delay = self.retry_backoff * (2 ** attempt)
                        delay += random.uniform(0, delay * 0.25)
                        log.warning("MCP %s transient error: %s. Retrying in %.2fs",
                                    mcp_tool.name, repr(e), delay)
                        time.sleep(delay)
                    except Exception as e:  # non-transient, bail
                        error = str(e)
                        log.exception("MCP %s fatal error", mcp_tool.name)
                        break

                if document_tool_output is None and last_exc and error is None:
                    error = f"Transient error after {self.max_retries} retries: {last_exc}"

                if document_tool_output is not None:
                    # Format the tool output according to the tool's parser
                    text_output = mcp_tool._format_output(document_tool_output)
                    text_output = mcp_tool.tool_parser.format_result(text_output, document_tool_output)
                    found_tool = True
                    break

        except Exception as e:
            error = str(e)
            log.exception("MCP wrapper error")

        if document_tool_output is None:
            msg = error or "No valid tool calls found."
            return ToolOutput(
                output=msg,
                called=False,
                error=msg,
                timeout=False,
                runtime=0,
                start_str="<snippet id=" + generate_snippet_id() + ">\n",
                end_str="\n</snippet>",
            )

        if document_tool_output.error:
            log.error("Error from mcp tool: %s", document_tool_output.error)

        log.debug("Tool call prompt (truncated): %s", trunc_prompt)
        log.debug("Returning tool output: %s", text_output)

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
    from open_instruct.grpo_fast import launch_mcp_subprocess
    import time as _t

    launch_mcp_subprocess(
        "uv run python -m rl-rag-mcp.mcp_agents.mcp_backend.main --transport http --port 8000 --host 0.0.0.0 --path /mcp",
        "./mcp_logs",
    )
    # Poll instead of blind sleep
    _t.sleep(6)

    mcp_tool = MCPTool(["google_search"], number_documents_to_search=10, api_endpoint="http://localhost:8000/mcp")
    print(mcp_tool('<tool name="google_search">What is the capital of France?</tool>'))