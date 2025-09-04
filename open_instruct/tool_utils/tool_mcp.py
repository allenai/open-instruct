import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional, Union

from fastmcp import Client
from mcp.types import CallToolResult

from open_instruct.search_rewards.utils.format_utils import generate_snippet_id
from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput


class MCPTool(Tool):
    def __init__(self, start_str: str, end_str: str, timeout: int = 60):
        self.start_str = start_str
        self.end_str = end_str
        self.timeout = timeout  # to avoid hanging forever

    def init_mcp_client(self):
        if os.environ.get("MCP_TRANSPORT") == "StreamableHttpTransport":
            port = os.environ.get("MCP_TRANSPORT_PORT", 8000)
            print(
                f"Using MCP transport: {os.environ.get('MCP_TRANSPORT')}, port: {port}"
            )
            return Client(f"http://localhost:{port}/mcp", timeout=self.timeout)
        elif os.environ.get("MCP_TRANSPORT") == "FastMCPTransport":
            mcp_executable = os.environ.get("MCP_EXECUTABLE")
            print(
                f"Using MCP transport: {os.environ.get('MCP_TRANSPORT')}, executable: {mcp_executable}"
            )
            return Client(mcp_executable, timeout=self.timeout)
        else:
            raise ValueError(
                f"Invalid MCP transport: {os.environ.get('MCP_TRANSPORT')}"
            )

    def _find_query_blocks(self, prompt: str) -> list[str]:
        # Find Python code blocks using regex
        re_str = r"<tool>\s*(.*?)\s*</tool>"  # we replace <tool> immediately with the custom start_str
        re_str = re_str.replace("<tool>", self.start_str).replace(
            "</tool>", self.end_str
        )

        query_blocks = re.findall(re_str, prompt, re.DOTALL)
        return query_blocks

    def __call__(self, prompt: str) -> ToolOutput:
        raise NotImplementedError("Subclasses must implement this method")


class SemanticScholarSnippetSearchTool(MCPTool):
    def __init__(self, start_str: str, end_str: str, *args, **kwargs):
        self.mcp_tool_name = "semantic_scholar_snippet_search"
        self.number_documents_to_search = kwargs.pop("number_documents_to_search", 3)
        print(f"SemanticScholarSnippetSearchTool initialized with mcp_tool_name: {self.mcp_tool_name}, number_documents_to_search: {self.number_documents_to_search}")
        super().__init__(start_str, end_str, *args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        print(f"SemanticScholarSnippetSearchTool called with prompt: {prompt}")
        mcp_client = self.init_mcp_client()
        query_blocks = self._find_query_blocks(prompt)
        print(f"Query blocks: {query_blocks}")

        if len(query_blocks) == 0:
            return ToolOutput(
                output="",
                called=False,
                error="No valid query found in tool call.",
                timeout=False,
                runtime=0,
                start_str="<snippet id=x>\n",
                end_str="\n</snippet>",
            )

        # Only execute the last code block
        query_string = query_blocks[-1]
        if not query_string:
            return ToolOutput(
                output="Empty query. Please provide some text in the query.",
                error="Empty query. Please provide some text in the query.",
                called=True,
                timeout=False,
                runtime=0,
                start_str="<snippet id=x>\n",
                end_str="\n</snippet>",
            )

        start_time = time.time()
        timeout = False
        error = ""

        async def async_search(query: str, tool_name: str, limit: int):
            try:
                async with mcp_client:
                    await mcp_client.ping()
                    # Call the MCP tool with the query
                    result = await mcp_client.call_tool(
                        tool_name,
                        {"query": query, "limit": limit},
                    )
                    return json.loads(result.content[0].text)
            except Exception as e:
                return {"error": str(e), "content": []}

        # for now, we just join all snippets.
        output = asyncio.run(
            async_search(
                query_string,
                self.mcp_tool_name,
                self.number_documents_to_search,
            )
        )

        if (
            not output
            or (error := output.get("error"))
            or not output.get("data")
            or len(output["data"]) == 0
        ):
            print(f"Query failed for unknown reason: {error}")
            print(f"Output: {output}")
            return ToolOutput(
                output="Query failed with error.",
                error=f"Query failed for unknown reason: {error}",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
                start_str="<snippet id=x>\n",
                end_str="\n</snippet>",
            )

        snippets = [ele["snippet"]["text"].strip() for ele in output["data"]]
        all_snippets = "\n".join(snippets).strip()
        print(f"All snippets: {all_snippets}")

        return ToolOutput(
            output=all_snippets,
            called=True,
            error=error,
            timeout=timeout,
            runtime=time.time() - start_time,
            start_str="<snippet id=" + generate_snippet_id() + ">\n",
            end_str="\n</snippet>",
        )


class SerperSearchTool(MCPTool):
    def __init__(self, start_str: str, end_str: str, *args, **kwargs):
        self.mcp_tool_name = "serper_google_webpage_search"
        self.number_documents_to_search = kwargs.pop("number_documents_to_search", 5)

        super().__init__(start_str, end_str, *args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        mcp_client = self.init_mcp_client()
        query_blocks = self._find_query_blocks(prompt)

        if len(query_blocks) == 0:
            return ToolOutput(
                output="",
                called=False,
                error="",
                timeout=False,
                runtime=0,
            )

        # Only execute the last code block
        query_string = query_blocks[-1]
        if not query_string:
            return ToolOutput(
                output="",
                error="Empty query. Please provide some text in the query.",
                called=True,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()
        timeout = False
        error = ""

        async def async_search(query: str, tool_name: str, limit: int):
            try:
                async with mcp_client:
                    await mcp_client.ping()
                    # Call the MCP tool with the query
                    result = await mcp_client.call_tool(
                        tool_name,
                        {"query": query, "num_results": limit},
                    )
                    return json.loads(result.content[0].text)
            except Exception as e:
                return {"error": str(e), "content": []}

        # for now, we just join all snippets.
        output = asyncio.run(
            async_search(
                query_string,
                self.mcp_tool_name,
                self.number_documents_to_search,
            )
        )

        if (
            not output
            or (error := output.get("error"))
            or not output.get("organic")
            or len(output["organic"]) == 0
        ):
            print(f"Query failed for unknown reason: {error}")
            print(f"Output: {output}")
            return ToolOutput(
                output="",
                error=f"Query failed for unknown reason: {error}",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )

        snippets = [ele["snippet"].strip() for ele in output["organic"]]
        all_snippets = "\n".join(snippets).strip()

        return ToolOutput(
            output=all_snippets,
            called=True,
            error=error,
            timeout=timeout,
            runtime=time.time() - start_time,
            start_str="<snippet id=" + generate_snippet_id() + ">\n",
            end_str="\n</snippet>",
        )


# Simple test for the MCP tool
if __name__ == "__main__":

    from concurrent.futures import ThreadPoolExecutor

    tool1 = SemanticScholarSnippetSearchTool(start_str="<search>", end_str="</search>")
    tool2 = SerperSearchTool(start_str="<search>", end_str="</search>")

    with ThreadPoolExecutor(max_workers=20) as executor:
        future1 = executor.submit(
            tool1,
            "<search>what are the primary source of model hallucinations?</search>",
        )
        future2 = executor.submit(
            tool2,
            "<search>what are the primary source of model hallucinations?</search>",
        )
        result1 = future1.result()
        result2 = future2.result()

    print("Test SemanticScholarSnippetSearchTool result:", result1)
    print("Test SerperSearchTool result:", result2)
