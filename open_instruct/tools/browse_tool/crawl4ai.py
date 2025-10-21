import os
from typing import Optional, List

from transformers.models.esm.openfold_utils import data_transforms

from open_instruct.tools.search_tool.s2 import create_session_with_retries


def _get_bool_env(var_name: str, default: bool) -> bool:
    val = os.environ.get(var_name)
    if val is None:
        return default
    val_lower = val.strip().lower()
    return val_lower in {"1", "true", "yes", "y", "on"}


def _load_blocklist_from_env() -> Optional[List[str]]:
    """
    Load a newline-delimited domain blocklist from CRAWL4AI_BLOCKLIST_PATH.
    Falls back to bundled crawl4ai_blocklist.txt if present.
    Returns None if no blocklist can be found.
    """
    # Env override first
    blocklist_path = os.environ.get("CRAWL4AI_BLOCKLIST_PATH")
    candidates: List[str] = []

    if blocklist_path and os.path.exists(blocklist_path):
        candidates.append(blocklist_path)

    # Fallback to the bundled blocklist in this package directory
    bundled_path = os.path.join(os.path.dirname(__file__), "crawl4ai_blocklist.txt")
    if os.path.exists(bundled_path):
        candidates.append(bundled_path)

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                # Deduplicate while preserving order
                seen = set()
                unique_lines = []
                for line in lines:
                    if line not in seen:
                        seen.add(line)
                        unique_lines.append(line)
                return unique_lines or None
        except Exception:
            # If one candidate fails, try next
            continue

    return None


def crawl_url(url: str, api_endpoint: str = None, timeout: int = 60) -> Optional[str]:
    """
    Crawl a single URL and return its markdown content.

    This is a simplified synchronous wrapper that uses the Crawl4AI Docker API.
    For more advanced features (BM25 filtering, pruning, AI2 config), use the async version.

    Args:
        url: Target URL to crawl
        api_endpoint: Base URL for the Crawl4AI Docker API (if not provided, uses CRAWL4AI_API_URL env var)
        timeout: Timeout in seconds (default: 60)

    Returns:
        String containing the markdown content of the page, or None if crawl failed
    """
    if not api_endpoint:
        api_endpoint = os.environ.get("CRAWL4AI_API_URL")
        if not api_endpoint:
            raise ValueError("CRAWL4AI_API_URL environment variable is not set or api_endpoint parameter not provided")

    api_key = os.environ.get("CRAWL4AI_API_KEY")

    session = create_session_with_retries()

    # Build the request payload, forcing AI2 config usage
    use_ai2_config = True
    bypass_cache = _get_bool_env("CRAWL4AI_BYPASS_CACHE", True)
    ignore_links = _get_bool_env("CRAWL4AI_IGNORE_LINKS", True)
    include_html = _get_bool_env("CRAWL4AI_INCLUDE_HTML", False)
    use_pruning = _get_bool_env("CRAWL4AI_USE_PRUNING", False)
    check_robots = _get_bool_env("CRAWL4AI_CHECK_ROBOTS", True)
    user_agent = os.environ.get("CRAWL4AI_USER_AGENT")
    if not user_agent and use_ai2_config:
        # Default to AI2 bot UA if using AI2 config and no explicit UA override
        user_agent = "Mozilla/5.0 (compatible) AI2Bot-DeepResearchEval (+https://www.allenai.org/crawler)"
    page_timeout_ms_str = os.environ.get("CRAWL4AI_PAGE_TIMEOUT_MS")
    query = os.environ.get("CRAWL4AI_QUERY")

    payload = {"urls": [url], "bypass_cache": bypass_cache, "ignore_links": ignore_links}
    payload["use_ai2_config"] = True

    # Optional payload fields (only include when set to avoid strict server schemas)
    if include_html:
        payload["include_html"] = True
    if use_pruning:
        payload["use_pruning"] = True
    if query:
        payload["query"] = query
    if user_agent:
        payload["user_agent"] = user_agent
    if check_robots is not None:
        payload["check_robots_txt"] = check_robots
    if page_timeout_ms_str:
        try:
            payload["page_timeout"] = int(page_timeout_ms_str)
        except ValueError:
            pass

    # Domain blocklist support
    exclude_domains = _load_blocklist_from_env()
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["x-api-key"] = api_key

    # Call the Crawl4AI Docker API
    response = session.post(f"{api_endpoint}/crawl", json=payload, headers=headers, timeout=timeout)
    response.raise_for_status()

    data = response.json()
    if not data.get("success", False):
        error_msg = data.get("error_message") or data.get("message") or "Unknown error"
        print(f"Error crawling {url}: {error_msg}")
        return None

    results = data["results"]

    result = results[0] if isinstance(results, list) else results
    markdown = result.get("markdown", {})
    fit_markdown_content = markdown.get("fit_markdown", "")
    raw_markdown_content = markdown.get("raw_markdown", "")
    html_content = result.get("html", "")

    final_content = fit_markdown_content or raw_markdown_content or html_content

    if not final_content:
        print(f"Warning: No content returned for {url}")
        final_content = "No content extracted from the page"

    return final_content
