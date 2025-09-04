import re
import time

from open_instruct.search_rewards.utils.format_utils import generate_snippet_id

from open_instruct.search_utils.massive_ds import get_snippets_for_query as get_snippets_for_query_massive_ds
from open_instruct.search_utils.s2 import get_snippets_for_query as get_snippets_for_query_s2
from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput


class SearchTool(Tool):
    def __init__(self, api_endpoint: str, use_massive_ds: bool = False, *args, **kwargs):
        self.api_endpoint = api_endpoint
        if use_massive_ds:
            self.get_snippets_for_query = get_snippets_for_query_massive_ds
        else:
            self.get_snippets_for_query = get_snippets_for_query_s2
        self.start_str = "<search>"
        self.end_str = "</search>"
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
        snippets = self.get_snippets_for_query(
            query_string, api_endpoint=self.api_endpoint, number_of_results=self.number_documents_to_search
        )

        if not snippets or len(snippets) == 0:
            return ToolOutput(
                output="Query failed for unknown reason.",
                error="Query failed for unknown reason.",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
                start_str="<snippet id=x>\n",
                end_str="\n</snippet>",
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
            start_str="<snippets id=" + generate_snippet_id() + ">\n",
            end_str="\n</snippets>",
        )
