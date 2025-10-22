import re
import os
import time
from typing import Callable, List

from open_instruct.tools.search_tool.massive_ds import get_snippets_for_query as massive_ds_get_snippets_for_query
from open_instruct.tools.search_tool.s2 import get_snippets_for_query as s2_get_snippets_for_query
from open_instruct.tools.search_tool.you import get_snippets_for_query as you_get_snippets_for_query
from open_instruct.tools.search_tool.serper import get_snippets_for_query as serper_get_snippets_for_query
from open_instruct.tools.utils.tool_classes import Tool, ToolOutput


class SearchTool(Tool):
    def __init__(self, snippet_fn: Callable[[str], List[str]], api_endpoint: str | None = None, *args, **kwargs):
        self.snippet_fn = snippet_fn
        self.api_endpoint = api_endpoint
        self.start_str = "<query>"
        self.end_str = "</query>"
        self.number_documents_to_search = kwargs.pop("number_documents_to_search", 3)
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        # Find Python code blocks using regex
        re_str = r"<tool>\s*(.*?)\s*</tool>"  # we replace <tool> immediately with the custom start_str
        re_str = re_str.replace("<tool>", self.start_str).replace("</tool>", self.end_str)

        query_blocks = re.findall(re_str, prompt, re.DOTALL)

        if len(query_blocks) == 0:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

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
        snippets = self.snippet_fn(
            query_string, api_endpoint=self.api_endpoint, number_of_results=self.number_documents_to_search
        )

        if not snippets or len(snippets) == 0:
            return ToolOutput(
                output="",
                error="Query failed for unknown reason.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )

        # for now, we just join all snippets.
        snippets = [snippet.strip() for snippet in snippets]
        all_snippets = "\n".join(snippets).strip()

        return ToolOutput(
            output=all_snippets,
            called=True,
            error=error,
            timeout=timeout,
            runtime=time.time() - start_time,
            start_str="<output>\n",
            end_str="\n</output>",
        )


class S2SearchTool(SearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(s2_get_snippets_for_query, *args, **kwargs)
        self.start_str = "<query_s2>"
        self.end_str = "</query_s2>"


class YouSearchTool(SearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(you_get_snippets_for_query, *args, **kwargs)
        self.start_str = "<query_you>"
        self.end_str = "</query_you>"


class MassiveDSSearchTool(SearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(massive_ds_get_snippets_for_query, *args, **kwargs)
        self.start_str = "<query_massive_ds>"
        self.end_str = "</query_massive_ds>"
        # If the MASSIVE_DS_API_URL environment variable is set, use it as the API endpoint
        if os.environ.get("MASSIVE_DS_API_URL") is not None:
            self.api_endpoint = os.environ.get("MASSIVE_DS_API_URL")


class SerperSearchTool(SearchTool):
    def __init__(self, *args, **kwargs):
        super().__init__(serper_get_snippets_for_query, *args, **kwargs)
        self.start_str = "<query_serper>"
        self.end_str = "</query_serper>"
