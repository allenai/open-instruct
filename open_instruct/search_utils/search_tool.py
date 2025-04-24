from open_instruct.tool_utils.tool_vllm import Tool, ToolOutput
from open_instruct.search_utils.massive_ds import get_snippets_for_query

class SearchTool(Tool):
    def __init__(self, api_endpoint: str, *args, **kwargs):
        self.api_endpoint = api_endpoint
        self.start_str = "<query>"
        self.end_str = "</query>"
        self.number_documents_to_search = kwargs.pop("number_documents_to_search", 3)
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        # Find Python code blocks using regex
        re_str = r'<tool>\s*(.*?)\s*</tool>' # we replace <tool> immediately with the custom start_str
        query_string = re_str.replace('<tool>', self.start_str).replace('</tool>', self.end_str)

        if not query_string:
            return ToolOutput(output="<document>Empty Query.</document>", called=False, success=False)
        
        snippets = get_snippets_for_query(query_string, api_endpoint=self.api_endpoint, number_of_results=self.number_documents_to_search)

        if not snippets or len(snippets) == 0:
            return ToolOutput(output="<document>Query failed.</document>", called=True, success=False)

        # for now, we just join all snippets.
        all_snippets = "\n".join(snippets)

        return ToolOutput(output=f"<document>{all_snippets}</document>", called=True, success=True)
