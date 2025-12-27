"""
Tools that follow the Tool(ABC) pattern.
Each tool has a corresponding Config dataclass and a from_config() classmethod.
"""

import asyncio
import inspect
import os
import time
import traceback
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from open_instruct.tools.base import Tool, ToolOutput

# Optional imports for MCP tools
try:
    import httpcore
    import httpx
    from dr_agent.tool_interface.mcp_tools import (
        Crawl4AIBrowseTool,
        MassiveServeSearchTool,
        SemanticScholarSnippetSearchTool,
        SerperSearchTool,
    )

    MCP_AVAILABLE = True
    MCP_TOOL_REGISTRY: dict[str, type] = {
        "snippet_search": SemanticScholarSnippetSearchTool,
        "google_search": SerperSearchTool,
        "massive_serve": MassiveServeSearchTool,
        "browse_webpage": Crawl4AIBrowseTool,
    }
except ImportError:
    MCP_AVAILABLE = False
    MCP_TOOL_REGISTRY = {}


def _create_session_with_retries(
    retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Collection[int] = (500, 502, 504),
    allowed_methods: Collection[str] = ("GET", "POST"),
) -> requests.Session:
    """Create a requests session with automatic retries."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


# =============================================================================
# MaxCallsExceededTool (no config needed)
# =============================================================================


class MaxCallsExceededTool(Tool):
    """Tool that returns a message when max tool calls have been exceeded."""

    tool_function_name = "max_calls_exceeded"
    tool_args: dict[str, dict[str, str]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> ToolOutput:
        return ToolOutput(output="Max tool calls exceeded.", called=False, error="", timeout=False, runtime=0)


# =============================================================================
# PythonCodeTool + Config
# =============================================================================


@dataclass
class PythonCodeToolConfig:
    """Configuration for the Python code execution tool."""

    api_endpoint: str | None = None
    """The API endpoint for the code execution server."""
    timeout_seconds: int = 3
    """Timeout in seconds for code execution."""


class PythonCodeTool(Tool):
    """
    Executes Python code via a FastAPI endpoint.

    @vwxyzjn: I recommend using something like a FastAPI for this kind of stuff; 1) you
    won't accidentally block the main vLLM process and 2) way easier to parallelize via load balancing.
    """

    tool_function_name = "python"
    tool_args: dict[str, dict[str, str]] = {"code": {"type": "string", "description": "Python code to execute"}}

    def __init__(self, api_endpoint: str, timeout_seconds: int = 3) -> None:
        self.api_endpoint = api_endpoint
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_config(cls, config: PythonCodeToolConfig) -> "PythonCodeTool":
        if not config.api_endpoint:
            raise ValueError("api_endpoint must be set to use the Python code tool")
        return cls(api_endpoint=config.api_endpoint, timeout_seconds=config.timeout_seconds)

    def __call__(self, code: str) -> ToolOutput:
        """Execute Python code via the API."""
        if not code or not code.strip():
            return ToolOutput(
                output="",
                error="Empty code. Please provide some code to execute.",
                called=True,
                timeout=False,
                runtime=0,
            )

        all_outputs = []
        timeout = False
        error = ""
        start_time = time.time()

        try:
            response = requests.post(
                self.api_endpoint, json={"code": code, "timeout": self.timeout_seconds}, timeout=self.timeout_seconds
            )
            result = response.json()
            output = result["output"]
            error = result.get("error") or ""

            all_outputs.append(output)
            if len(error) > 0:
                all_outputs.append("\n" + error)

        except requests.Timeout:
            all_outputs.append(f"Timeout after {self.timeout_seconds} seconds")
            timeout = True

        except Exception as e:
            error_message = f"Error calling API: {str(e)}\n"
            error_traceback = traceback.format_exc()
            all_outputs.append(error_message + error_traceback)

        return ToolOutput(
            output="\n".join(all_outputs), called=True, error=error, timeout=timeout, runtime=time.time() - start_time
        )


# =============================================================================
# SearchTool + Config
# =============================================================================


@dataclass
class SearchToolConfig:
    """Configuration for the massive_ds search tool."""

    api_endpoint: str | None = None
    """The API endpoint for the search engine."""
    number_documents: int = 3
    """The maximum number of documents to retrieve for each query."""


class SearchTool(Tool):
    """Search tool using the massive_ds API."""

    tool_function_name = "search"
    tool_args: dict[str, dict[str, str]] = {"query": {"type": "string", "description": "The search query"}}

    def __init__(self, api_endpoint: str | None = None, number_documents_to_search: int = 3) -> None:
        self.api_endpoint = api_endpoint
        self.number_documents_to_search = number_documents_to_search

    @classmethod
    def from_config(cls, config: SearchToolConfig) -> "SearchTool":
        return cls(api_endpoint=config.api_endpoint, number_documents_to_search=config.number_documents)

    def __call__(self, query: str) -> ToolOutput:
        """Search for documents matching the query."""
        if not query or not query.strip():
            return ToolOutput(
                output="",
                error="Empty query. Please provide some text in the query.",
                called=True,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()

        # Resolve API endpoint
        url = self.api_endpoint
        if not url:
            url = os.environ.get("MASSIVE_DS_URL")
            if not url:
                return ToolOutput(
                    output="",
                    error="Missing MASSIVE_DS_URL environment variable.",
                    called=True,
                    timeout=False,
                    runtime=time.time() - start_time,
                )

        session = _create_session_with_retries()

        try:
            res = session.post(
                url,
                json={"query": query, "n_docs": self.number_documents_to_search, "domains": "dpr_wiki_contriever"},
                headers={"Content-Type": "application/json"},
                timeout=(3, 15),
            )
            res.raise_for_status()
            data = res.json()
            passages = data.get("results", {}).get("passages", [[]])[0]
            passages = passages[: self.number_documents_to_search]
            passages = ["\n" + passage for passage in passages]
            all_snippets = "\n".join(passages).strip()

            return ToolOutput(
                output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
            )

        except requests.exceptions.RequestException as e:
            return ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)


# =============================================================================
# S2SearchTool + Config
# =============================================================================


@dataclass
class S2SearchToolConfig:
    """Configuration for the Semantic Scholar search tool."""

    number_of_results: int = 10
    """Number of results to return from Semantic Scholar."""


class S2SearchTool(Tool):
    """
    Search tool using the Semantic Scholar API.
    Requires S2_API_KEY environment variable.
    """

    tool_function_name = "s2_search"
    tool_args: dict[str, dict[str, str]] = {
        "query": {"type": "string", "description": "The search query for Semantic Scholar"}
    }

    def __init__(self, number_of_results: int = 10) -> None:
        self.number_of_results = number_of_results

    @classmethod
    def from_config(cls, config: S2SearchToolConfig) -> "S2SearchTool":
        return cls(number_of_results=config.number_of_results)

    def __call__(self, query: str) -> ToolOutput:
        """Search Semantic Scholar for documents matching the query."""
        if not query or not query.strip():
            return ToolOutput(
                output="",
                error="Empty query. Please provide some text in the query.",
                called=True,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()

        api_key = os.environ.get("S2_API_KEY")
        if not api_key:
            return ToolOutput(
                output="",
                error="Missing S2_API_KEY environment variable.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )

        session = _create_session_with_retries()

        try:
            res = session.get(
                "https://api.semanticscholar.org/graph/v1/snippet/search",
                params={"limit": self.number_of_results, "query": query},
                headers={"x-api-key": api_key},
                timeout=60,
            )
            res.raise_for_status()
            data = res.json().get("data", [])
            snippets = [item["snippet"]["text"] for item in data if item.get("snippet")]

            if not snippets:
                return ToolOutput(
                    output="",
                    error="Query returned no results.",
                    called=True,
                    timeout=False,
                    runtime=time.time() - start_time,
                )

            all_snippets = "\n".join(snippets).strip()
            return ToolOutput(
                output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
            )

        except requests.exceptions.RequestException as e:
            return ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)


# =============================================================================
# YouSearchTool + Config
# =============================================================================


@dataclass
class YouSearchToolConfig:
    """Configuration for the You.com search tool."""

    number_of_results: int = 10
    """Number of results to return from You.com."""


class YouSearchTool(Tool):
    """
    Search tool using the You.com API.
    Requires YOUCOM_API_KEY environment variable.
    """

    tool_function_name = "you_search"
    tool_args: dict[str, dict[str, str]] = {"query": {"type": "string", "description": "The search query for You.com"}}

    def __init__(self, number_of_results: int = 10) -> None:
        self.number_of_results = number_of_results

    @classmethod
    def from_config(cls, config: YouSearchToolConfig) -> "YouSearchTool":
        return cls(number_of_results=config.number_of_results)

    def __call__(self, query: str) -> ToolOutput:
        """Search You.com for documents matching the query."""
        if not query or not query.strip():
            return ToolOutput(
                output="",
                error="Empty query. Please provide some text in the query.",
                called=True,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()

        api_key = os.environ.get("YOUCOM_API_KEY")
        if not api_key:
            return ToolOutput(
                output="",
                error="Missing YOUCOM_API_KEY environment variable.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )

        session = _create_session_with_retries()

        try:
            response = session.get(
                "https://api.ydc-index.io/search",
                params={"query": query, "num_web_results": 1},
                headers={"X-API-Key": api_key},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if "error_code" in data:
                return ToolOutput(
                    output="",
                    error=f"API error: {data['error_code']}",
                    called=True,
                    timeout=False,
                    runtime=time.time() - start_time,
                )

            snippets = []
            for hit in data.get("hits", []):
                for snippet in hit.get("snippets", []):
                    snippets.append(snippet)

            snippets = snippets[: self.number_of_results]

            if not snippets:
                return ToolOutput(
                    output="",
                    error="Query returned no results.",
                    called=True,
                    timeout=False,
                    runtime=time.time() - start_time,
                )

            all_snippets = "\n".join(["\n" + snippet for snippet in snippets]).strip()
            return ToolOutput(
                output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
            )

        except requests.exceptions.RequestException as e:
            return ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)


# =============================================================================
# SerperSearchTool + Config
# =============================================================================


@dataclass
class SerperSearchToolConfig:
    """Configuration for the Serper (Google Search) tool."""

    number_of_results: int = 5
    """Number of results to return from Serper."""


class SerperSearchTool(Tool):
    """
    Search tool using the Serper API (Google Search results).
    Requires SERPER_API_KEY environment variable.

    Serper provides fast Google Search results via API. Sign up at https://serper.dev
    """

    tool_function_name = "serper_search"
    tool_args: dict[str, dict[str, str]] = {
        "query": {"type": "string", "description": "The search query for Google via Serper"}
    }

    def __init__(self, number_of_results: int = 5) -> None:
        self.number_of_results = number_of_results

    @classmethod
    def from_config(cls, config: SerperSearchToolConfig) -> "SerperSearchTool":
        return cls(number_of_results=config.number_of_results)

    def __call__(self, query: str) -> ToolOutput:
        """Search Google via Serper for documents matching the query."""
        if not query or not query.strip():
            return ToolOutput(
                output="",
                error="Empty query. Please provide some text in the query.",
                called=True,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()

        api_key = os.environ.get("SERPER_API_KEY")
        if not api_key:
            return ToolOutput(
                output="",
                error="Missing SERPER_API_KEY environment variable.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )

        session = _create_session_with_retries()

        try:
            response = session.post(
                "https://google.serper.dev/search",
                json={"q": query, "num": self.number_of_results},
                headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            snippets = []

            # Extract snippets from organic results
            for result in data.get("organic", [])[: self.number_of_results]:
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")
                if snippet:
                    snippets.append(f"**{title}**\n{snippet}\nSource: {link}")

            # Also include answer box if present
            if "answerBox" in data:
                answer_box = data["answerBox"]
                if "answer" in answer_box:
                    snippets.insert(0, f"**Direct Answer:** {answer_box['answer']}")
                elif "snippet" in answer_box:
                    snippets.insert(0, f"**Featured Snippet:** {answer_box['snippet']}")

            if not snippets:
                return ToolOutput(
                    output="",
                    error="Query returned no results.",
                    called=True,
                    timeout=False,
                    runtime=time.time() - start_time,
                )

            all_snippets = "\n\n".join(snippets).strip()
            return ToolOutput(
                output=all_snippets, called=True, error="", timeout=False, runtime=time.time() - start_time
            )

        except requests.exceptions.RequestException as e:
            return ToolOutput(output="", error=str(e), called=True, timeout=False, runtime=time.time() - start_time)


# =============================================================================
# MCPTool + Config
# =============================================================================


@dataclass
class MCPToolConfig:
    """Configuration for MCP (Model Context Protocol) tools."""

    tool_names: str = "snippet_search"
    """Comma-separated list of MCP tool names to use."""
    parser_name: str = "unified"
    """The parser name for MCP tools."""
    transport_type: str | None = None
    """Transport type for MCP (default: StreamableHttpTransport)."""
    host: str | None = None
    """Host for MCP transport."""
    port: int | None = None
    """Port for MCP transport."""
    timeout: int = 180
    """Timeout in seconds for MCP tool calls."""
    max_retries: int = 3
    """Maximum retries for transient MCP errors."""
    retry_backoff: float = 0.5
    """Backoff factor for MCP retries."""
    base_url: str | None = None
    """Base URL for MCP tools."""
    number_documents: int = 10
    """Number of documents to search for MCP tools."""
    use_localized_snippets: bool = False
    """Whether to use localized snippets."""
    context_chars: int = 6000
    """Number of context characters for MCP tools."""


class MCPTool(Tool):
    """
    A wrapper for MCP (Model Context Protocol) tools from dr_agent.

    Unlike other tools, this handles multiple MCP tools that share the same
    end string (</tool>). It routes calls to the appropriate underlying tool.

    Requires the dr_agent package to be installed.
    """

    tool_function_name = "mcp"
    tool_args: dict[str, dict[str, str]] = {
        "prompt": {"type": "string", "description": "The full prompt containing MCP tool calls"}
    }

    def __init__(
        self,
        mcp_tool_names: list[str] | str,
        mcp_parser_name: str = "unified",
        transport_type: str | None = None,
        mcp_host: str | None = None,
        mcp_port: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 0.5,
        mcp_timeout: int = 180,
        base_url: str | None = None,
        number_documents_to_search: int = 10,
        use_localized_snippets: bool = False,
        context_chars: int = 6000,
        **kwargs: Any,
    ) -> None:
        if not MCP_AVAILABLE:
            raise ImportError("MCP tools require dr_agent package. Install it with: pip install dr_agent")

        self.mcp_tools: list[Any] = []
        self.stop_strings: list[str] = []
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Configure transport
        self.transport_type = transport_type or os.environ.get("MCP_TRANSPORT", "StreamableHttpTransport")
        self.mcp_host = mcp_host or os.environ.get("MCP_TRANSPORT_HOST", "0.0.0.0")
        self.mcp_port = mcp_port or os.environ.get("MCP_TRANSPORT_PORT", "8000")

        if self.mcp_host is not None:
            os.environ["MCP_TRANSPORT_HOST"] = str(self.mcp_host)
        if self.mcp_port is not None:
            os.environ["MCP_TRANSPORT_PORT"] = str(self.mcp_port)

        # Support comma-separated string for mcp_tool_names
        if isinstance(mcp_tool_names, str):
            mcp_tool_names = [n.strip() for n in mcp_tool_names.split(",") if n.strip()]

        for mcp_tool_name in mcp_tool_names:
            if mcp_tool_name not in MCP_TOOL_REGISTRY:
                raise ValueError(f"Unknown MCP tool: {mcp_tool_name}. Available: {list(MCP_TOOL_REGISTRY.keys())}")

            mcp_tool_cls = MCP_TOOL_REGISTRY[mcp_tool_name]
            sig = inspect.signature(mcp_tool_cls.__init__)
            valid_params = set(sig.parameters.keys())

            # Filter kwargs to only valid params
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
            if "base_url" in valid_params:
                filtered_kwargs["base_url"] = base_url
            if "number_documents_to_search" in valid_params:
                filtered_kwargs["number_documents_to_search"] = number_documents_to_search
            if "use_localized_snippets" in valid_params:
                filtered_kwargs["use_localized_snippets"] = use_localized_snippets
            if "context_chars" in valid_params:
                filtered_kwargs["context_chars"] = context_chars

            # Special config for crawl4ai
            if mcp_tool_name == "browse_webpage":
                filtered_kwargs["use_docker_version"] = True
                filtered_kwargs["use_ai2_config"] = True

            self.mcp_tools.append(
                mcp_tool_cls(
                    timeout=mcp_timeout,
                    name=mcp_tool_name,
                    tool_parser=mcp_parser_name,
                    transport_type=self.transport_type,
                    **filtered_kwargs,
                )
            )
            self.stop_strings += self.mcp_tools[-1].tool_parser.stop_sequences

    @classmethod
    def from_config(cls, config: MCPToolConfig, tool_name_override: str | None = None) -> "MCPTool":
        if not MCP_AVAILABLE:
            raise ImportError("MCP tools require dr_agent package. Install it with: pip install dr_agent")
        tool_names = tool_name_override if tool_name_override else config.tool_names
        return cls(
            mcp_tool_names=tool_names,
            mcp_parser_name=config.parser_name,
            transport_type=config.transport_type,
            mcp_host=config.host,
            mcp_port=config.port,
            mcp_timeout=config.timeout,
            max_retries=config.max_retries,
            retry_backoff=config.retry_backoff,
            base_url=config.base_url,
            number_documents_to_search=config.number_documents,
            use_localized_snippets=config.use_localized_snippets,
            context_chars=config.context_chars,
        )

    def get_stop_strings(self) -> list[str]:
        """Return the stop strings for all MCP tools."""
        return self.stop_strings

    def __call__(self, prompt: str) -> ToolOutput:
        """
        Execute the appropriate MCP tool based on the prompt content.

        Note: Unlike other tools, MCPTool still parses the prompt to determine
        which underlying tool to call, as MCP tools share common tags.
        """
        if not MCP_AVAILABLE:
            return ToolOutput(
                output="",
                error="MCP tools require dr_agent package. Install it with: pip install dr_agent",
                called=False,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()

        document_tool_output = None
        error = None
        found_tool = False
        text_output = ""

        try:
            for mcp_tool in self.mcp_tools:
                if mcp_tool.tool_parser.has_calls(prompt, mcp_tool.name):
                    # Retry on transient errors
                    for attempt in range(self.max_retries):
                        try:
                            document_tool_output = asyncio.run(mcp_tool(prompt))
                            break
                        except Exception as e:
                            # Check for transient network errors
                            if MCP_AVAILABLE and isinstance(
                                e, (httpcore.RemoteProtocolError, httpx.ReadError, ConnectionError, TimeoutError)
                            ):
                                if attempt + 1 >= self.max_retries:
                                    raise
                                time.sleep(self.retry_backoff * (2**attempt))
                            else:
                                raise

                    text_output = mcp_tool._format_output(document_tool_output)
                    text_output = mcp_tool.tool_parser.format_result(text_output, document_tool_output)
                    found_tool = True
                    break

        except Exception as e:
            error = str(e)

        if document_tool_output is None:
            if error is None and not found_tool:
                error = "No valid tool calls found."
            elif error is None:
                error = "Unknown error, no MCP response and no error found."

            return ToolOutput(
                output=error or "", called=False, error=error or "", timeout=False, runtime=time.time() - start_time
            )

        return ToolOutput(
            output=text_output,
            called=True,
            error=document_tool_output.error or "",
            timeout=document_tool_output.timeout,
            runtime=document_tool_output.runtime,
        )
