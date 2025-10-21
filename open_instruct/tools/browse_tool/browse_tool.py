import re
import time
from typing import Callable, Optional

from open_instruct.tools.utils.tool_classes import Tool, ToolOutput

from open_instruct.tools.browse_tool.crawl4ai import crawl_url as crawl4ai_crawl_url


class BrowseTool(Tool):
    def __init__(self, crawl_fn: Callable[[str], Optional[str]], api_endpoint: str, max_context_chars: int = 2000, *args, **kwargs):
        self.crawl_fn = crawl_fn
        self.api_endpoint = api_endpoint
        self.start_str = "<url>"
        self.end_str = "</url>"
        self.max_context_chars = max_context_chars
        super().__init__(*args, **kwargs)

    def __call__(self, prompt: str) -> ToolOutput:
        # Find URL blocks using regex
        re_str = r"<tool>\s*(.*?)\s*</tool>"  # we replace <tool> immediately with the custom start_str
        re_str = re_str.replace("<tool>", self.start_str).replace("</tool>", self.end_str)

        url_blocks = re.findall(re_str, prompt, re.DOTALL)

        if len(url_blocks) == 0:
            return ToolOutput(output="", called=False, error="", timeout=False, runtime=0)

        # Only execute the last URL block
        url_string = url_blocks[-1].strip()

        if not url_string:
            return ToolOutput(
                output="", error="Empty URL. Please provide a valid URL.", called=True, timeout=False, runtime=0
            )

        # Validate URL format (basic check)
        if not url_string.startswith(("http://", "https://")):
            return ToolOutput(
                output="",
                error=f"Invalid URL format: {url_string}. URL must start with http:// or https://",
                called=True,
                timeout=False,
                runtime=0,
            )

        start_time = time.time()
        timeout = False
        error = ""

        markdown_content = self.crawl_fn(url_string, api_endpoint=self.api_endpoint)

        if not markdown_content:
            return ToolOutput(
                output="",
                error=f"Failed to crawl URL: {url_string}",
                called=True,
                timeout=False,
                runtime=time.time() - start_time,
            )

        markdown_content = markdown_content.strip()
        if len(markdown_content) > self.max_context_chars:
            markdown_content = markdown_content[:self.max_context_chars]
            markdown_content += "..."

        # Return the markdown content
        return ToolOutput(
            output=markdown_content,
            called=True,
            error=error,
            timeout=timeout,
            runtime=time.time() - start_time,
            start_str="<webpage>\n",
            end_str="\n</webpage>",
        )


class Crawl4aiBrowseTool(BrowseTool):
    def __init__(self, *args, **kwargs):
        super().__init__(crawl4ai_crawl_url, *args, **kwargs)
        self.start_str = "<url_crawl4ai>"
        self.end_str = "</url_crawl4ai>"


if __name__ == "__main__":
    # Minimal quick test for manual running
    import os

    # Prefer env var for endpoint; this will raise in crawl4ai_crawl_url if unset
    api_endpoint = os.environ.get("CRAWL4AI_API_URL")

    tool = Crawl4aiBrowseTool(api_endpoint=api_endpoint, start_str="<url>", end_str="</url>")

    prompt = (
        "Here is a test. Please fetch this page.\n"
        "<url_crawl4ai>https://ivison.id.au</url_crawl4ai>\n"
    )

    out = tool(prompt)
    print("called:", out.called)
    print("error:", out.error)
    print("runtime:", round(out.runtime, 3), "s")
    if out.output:
        preview = out.output
        print("output preview:")
        print(preview)
