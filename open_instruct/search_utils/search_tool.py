import re
import time
from typing import Any

from open_instruct.search_utils.massive_ds import get_snippets_for_query
from open_instruct.tool_utils.tools import Tool, ToolOutput


class SearchTool(Tool):
    def __init__(self, api_endpoint: str, *args: Any, **kwargs: Any) -> None:
        self.api_endpoint = api_endpoint
        self.start_str = "<query>"
        self.end_str = "</query>"
        self.number_documents_to_search: int = kwargs.pop("number_documents_to_search", 3)
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
        snippets = get_snippets_for_query(
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
            start_str="<document>\n",
            end_str="\n</document>",
        )
