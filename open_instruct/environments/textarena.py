"""
Adapter for TextArena environments via OpenEnv.

TextArena provides games like Wordle through the OpenEnv standard.
Run server with: TEXTARENA_ENV_ID=Wordle-v0 uvicorn textarena_env.server.app:app
"""

import asyncio
import atexit
import os
import subprocess
import time
from typing import Any

import aiohttp

try:
    from textarena_env import TextArenaAction
    from textarena_env import TextArenaEnv as TextArenaClient

    _TEXTARENA_AVAILABLE = True
except ImportError:
    _TEXTARENA_AVAILABLE = False

from open_instruct.environments.base import RLEnvironment, StepResult
from open_instruct.utils import logger_utils

logger = logger_utils.setup_logger(__name__)


class TextArenaEnv(RLEnvironment):
    """Adapter for TextArena environments accessed via HTTP server.

    Wraps the TextArena client to provide the RLEnvironment interface.
    """

    def __init__(self, base_url: str = "", server_url: str = "", **kwargs: Any):
        """Initialize TextArena client.

        Args:
            base_url: URL of the TextArena server (e.g., "http://localhost:8000")
            server_url: Alternative parameter name for base_url (for pool compatibility)
            **kwargs: Additional kwargs (unused, for compatibility)
        """
        if not _TEXTARENA_AVAILABLE:
            raise ImportError("textarena-env required. Install with: uv pip install textarena-env")
        # Accept either base_url or server_url (server_url takes precedence for pool mode)
        self._base_url = server_url or base_url
        if not self._base_url:
            raise ValueError("Must provide either base_url or server_url")
        self._client: TextArenaClient | None = None
        self._last_observation: str = ""

    def _ensure_client(self):
        """Lazily create and connect the client when needed."""
        if self._client is None:
            self._client = TextArenaClient(base_url=self._base_url)

    def reset(self) -> StepResult:
        """Reset environment and get initial observation."""
        self._ensure_client()
        result = self._client.reset()
        # Extract the game prompt/instructions
        self._last_observation = result.observation.prompt
        return StepResult(
            observation=self._last_observation,
            reward=result.reward,
            done=result.done,
            info={"messages": [], "current_player_id": result.observation.current_player_id},
        )

    async def step(self, action: dict[str, Any]) -> StepResult:
        """Execute action (submit a guess/message to the game).

        Args:
            action: Action dict with "word" or "message" key (e.g., {"word": "crane"})
        """
        self._ensure_client()
        # Extract message from action dict
        word = action.get("word") or action.get("message", "")

        # TextArena Wordle expects words wrapped in square brackets
        message = f"[{word}]" if word and not word.startswith("[") else word

        # TextArena expects action wrapped in TextArenaAction
        result = self._client.step(TextArenaAction(message=message))

        # Extract observation from messages
        messages_content = []
        for msg in result.observation.messages:
            messages_content.append(msg.content)

        # Use the last message as the observation
        observation = messages_content[-1] if messages_content else ""

        return StepResult(
            observation=observation,
            reward=result.reward,
            done=result.done,
            info={"messages": messages_content, "current_player_id": result.observation.current_player_id},
        )

    def close(self):
        """Close the TextArena client."""
        if self._client:
            self._client.close()


class TextArenaServerManager:
    """Manages multiple TextArena server instances for parallel environments.

    Similar to AppWorldServerManager, this starts multiple uvicorn servers on
    different ports to support concurrent game sessions.
    """

    def __init__(self, pool_size: int, env_id: str = "Wordle-v0", base_port: int = 8765):
        """Initialize server manager.

        Args:
            pool_size: Number of server instances to start.
            env_id: TextArena environment ID (e.g., "Wordle-v0").
            base_port: Starting port number (subsequent servers use base_port+1, base_port+2, etc).
        """
        if not _TEXTARENA_AVAILABLE:
            raise ImportError("textarena-env required. Install with: uv pip install textarena-env")
        self.pool_size = pool_size
        self.env_id = env_id
        self.base_port = base_port
        self._processes: list[subprocess.Popen] = []
        self._server_urls: list[str] = []

    async def start(self) -> list[str]:
        """Start multiple TextArena server instances.

        Returns:
            List of server URLs (e.g., ["http://localhost:8765", "http://localhost:8766", ...])
        """
        logger.info(f"Starting {self.pool_size} TextArena servers for {self.env_id}...")

        for i in range(self.pool_size):
            port = self.base_port + i
            url = f"http://localhost:{port}"

            # Start uvicorn server in background
            env = os.environ.copy()
            env["TEXTARENA_ENV_ID"] = self.env_id

            proc = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--extra",
                    "openenv",
                    "python",
                    "-m",
                    "uvicorn",
                    "textarena_env.server.app:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(port),
                ],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            self._processes.append(proc)
            self._server_urls.append(url)

            # Add small delay every 10 servers to avoid overwhelming system
            if (i + 1) % 10 == 0:
                await asyncio.sleep(0.1)

        logger.info(f"All {self.pool_size} server processes started, waiting for health checks...")

        # Wait for all servers to be ready
        await self._wait_for_servers()
        logger.info(f"All {self.pool_size} TextArena servers ready!")

        # Register cleanup via atexit as fallback
        atexit.register(self.stop)

        return self._server_urls

    async def _wait_for_servers(self, timeout: float = 120.0):
        """Wait for all servers to respond to health checks.

        Args:
            timeout: Maximum time to wait in seconds (default: 120s for large pools)
        """
        start_time = time.time()
        ready = [False] * len(self._server_urls)
        last_report_time = start_time

        async with aiohttp.ClientSession() as session:
            while not all(ready) and (time.time() - start_time) < timeout:
                for i, url in enumerate(self._server_urls):
                    if ready[i]:
                        continue
                    try:
                        async with session.get(f"{url}/health", timeout=2.0) as resp:
                            if resp.status == 200:
                                ready[i] = True
                    except Exception:
                        pass

                # Progress report every 5 seconds
                current_time = time.time()
                if current_time - last_report_time >= 5.0:
                    ready_count = sum(ready)
                    elapsed = current_time - start_time
                    logger.info(
                        f"Health check progress: {ready_count}/{self.pool_size} servers ready ({elapsed:.1f}s elapsed)"
                    )
                    last_report_time = current_time

                if not all(ready):
                    await asyncio.sleep(0.5)

        ready_count = sum(ready)
        if not all(ready):
            self.stop()
            raise RuntimeError(
                f"Only {ready_count}/{self.pool_size} servers ready after {timeout}s. "
                f"Consider increasing timeout or reducing pool_size."
            )

    def stop(self):
        """Stop all TextArena server instances."""
        logger.info(f"Stopping {len(self._processes)} TextArena servers...")
        for proc in self._processes:
            proc.terminate()
        for proc in self._processes:
            proc.wait(timeout=5)
        self._processes = []
        self._server_urls = []

    @property
    def server_urls(self) -> list[str]:
        """Get list of server URLs."""
        return self._server_urls


# Module-level manager instance for cleanup.
_server_manager: TextArenaServerManager | None = None


async def setup_textarena_servers(pool_size: int, base_port: int = 8765) -> list[str]:
    """Return URLs for pre-started TextArena servers.

    This setup function assumes servers are already running (started externally,
    e.g., in the shell script before Ray initialization). It simply returns the
    expected URLs for the pool to connect to.

    Args:
        pool_size: Number of server instances
        base_port: Starting port number (default: 8765)

    Returns:
        List of server URLs for pre-started servers

    Note:
        Servers should be started before training with:
        ```bash
        for i in $(seq 0 $((POOL_SIZE-1))); do
            PORT=$((BASE_PORT + i))
            TEXTARENA_ENV_ID=Wordle-v0 uvicorn textarena_env.server.app:app --port ${PORT} &
        done
        ```
    """
    # Return URLs for pre-started servers
    urls = [f"http://localhost:{base_port + i}" for i in range(pool_size)]
    logger.info(f"Using {pool_size} pre-started TextArena servers: {urls[0]} to {urls[-1]}")
    return urls


def teardown_textarena_servers():
    """Explicitly shut down all TextArena servers.

    Call this at end of training if you want to clean up before process exit.
    """
    global _server_manager
    if _server_manager is not None:
        _server_manager.stop()
        _server_manager = None
