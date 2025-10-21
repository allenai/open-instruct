import os
from dataclasses import dataclass, field
from typing import Optional, List
from urllib.parse import urlparse

import crawl4ai
from crawl4ai import BrowserConfig, CacheMode, CrawlerRunConfig
from crawl4ai.docker_client import Crawl4aiDockerClient
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator


class Crawl4aiApiClient(Crawl4aiDockerClient):
    """Extended Docker client with custom path handling and authentication."""

    def __init__(self, base_url: str, *args, **kwargs):
        super().__init__(base_url=base_url, *args, **kwargs)
        self._path_url = urlparse(base_url).path

    async def _check_server(self) -> None:
        """Check if server is reachable."""
        await self._http_client.get(f"{self.base_url}{self._path_url}/health")
        self.logger.success(f"Connected to {self.base_url}", tag="READY")

    async def authenticate(self, api_key: str) -> None:
        """Set API key in headers."""
        self._http_client.headers["x-api-key"] = api_key

    def _request(self, method: str, endpoint: str, **kwargs):
        """Override request to add path prefix."""
        return super()._request(method, self._path_url + endpoint, **kwargs)


@dataclass(frozen=True)
class Ai2BotConfig:
    """AI2 bot configuration for Crawl4AI with blocklist and custom settings."""

    base_url: Optional[str] = field(default_factory=lambda: os.getenv("CRAWL4AI_API_URL"))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("CRAWL4AI_API_KEY"))
    blocklist_path: Optional[str] = field(default_factory=lambda: os.getenv("CRAWL4AI_BLOCKLIST_PATH"))

    user_agent: str = "Mozilla/5.0 (compatible) AI2Bot-DeepResearchEval (+https://www.allenai.org/crawler)"
    headless: bool = True
    browser_mode: str = "dedicated"
    use_managed_browser: bool = False
    user_agent_mode: str = ""
    user_agent_generator_config: dict = field(default_factory=lambda: {})
    extra_args: list = field(default_factory=lambda: [])
    enable_stealth: bool = False
    check_robots_txt: bool = True
    semaphore_count: int = 50

    def get_exclude_domains(self) -> list:
        if self.blocklist_path is None:
            raise ValueError(
                "CRAWL4AI_BLOCKLIST_PATH is not set; "
                "download the latest from https://github.com/allenai/crawler-rules/blob/main/blocklist.txt"
            )
        if not os.path.exists(self.blocklist_path):
            raise FileNotFoundError(f"Blocklist file not found: {self.blocklist_path}")
        with open(self.blocklist_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def get_browser_config(self, *args, **kwargs) -> BrowserConfig:
        return BrowserConfig(
            *args,
            headless=self.headless,
            user_agent=self.user_agent,
            browser_mode=self.browser_mode,
            use_managed_browser=self.use_managed_browser,
            user_agent_mode=self.user_agent_mode,
            user_agent_generator_config=self.user_agent_generator_config,
            extra_args=self.extra_args,
            enable_stealth=self.enable_stealth,
            **kwargs,
        )

    def get_crawler_config(self, *args, **kwargs) -> CrawlerRunConfig:
        return CrawlerRunConfig(
            *args,
            check_robots_txt=self.check_robots_txt,
            exclude_domains=self.get_exclude_domains(),
            geolocation=None,
            timezone_id=None,
            locale=None,
            simulate_user=False,
            semaphore_count=self.semaphore_count,
            user_agent=self.user_agent,
            user_agent_mode=self.user_agent_mode,
            user_agent_generator_config=self.user_agent_generator_config,
            **kwargs,
        )

    def get_base_url(self) -> str:
        if self.base_url is None:
            raise ValueError("CRAWL4AI_API_URL is not set")
        return self.base_url

    def get_api_key(self) -> str:
        if self.api_key is None:
            raise ValueError("CRAWL4AI_API_KEY is not set")
        return self.api_key


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


async def crawl_url(url: str, api_endpoint: str = None, timeout: int = 60000) -> Optional[str]:
    """
    Crawl a single URL and return its markdown content (or None on failure).

    Args:
        url: Target URL to crawl
        api_endpoint: Optional override for the Crawl4AI Docker API base URL
        timeout: Page timeout in milliseconds (defaults to 60000)

    Returns:
        Markdown string if successful, otherwise None
    """
    # Set up markdown generator with content filter and options
    md_generator = DefaultMarkdownGenerator(options={"ignore_links": True})

    # Build the request payload, forcing AI2 config usage
    ai2_config = Ai2BotConfig()
    api_key = ai2_config.get_api_key()
    base_url = ai2_config.get_base_url()
    browser_config = ai2_config.get_browser_config()
    crawler_config = ai2_config.get_crawler_config(
        cache_mode=CacheMode.BYPASS, markdown_generator=md_generator, page_timeout=timeout
    )

    async with Crawl4aiApiClient(base_url=base_url, verbose=False) as client:
        if api_key:
            await client.authenticate(api_key)

        # Crawl the URL
        results = await client.crawl([url], browser_config=browser_config, crawler_config=crawler_config)

        if not results:
            return None

        # Handle single result
        result = results[0] if isinstance(results, list) else results

        if not result.success:
            return None

        final_output = result.markdown.fit_markdown or result.markdown.raw_markdown or result.html
        return final_output or None


if __name__ == "__main__":
    import asyncio

    url = "https://ivison.id.au"
    print(asyncio.run(crawl_url(url)))
