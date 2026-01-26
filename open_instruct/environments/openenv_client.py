"""
Client adapter for external OpenEnv-compliant servers.

Connects to remote environments that expose the OpenEnv API via HTTP/WebSocket.
"""

from typing import Any

import aiohttp

from open_instruct.environments.base import RLEnvironment, StepResult


class OpenEnvClient(RLEnvironment):
    """Client for connecting to external OpenEnv-compliant servers.

    Communicates with remote environments via HTTP REST API.
    """

    def __init__(self, base_url: str, env_id: str | None = None, timeout: int = 30, **env_kwargs: Any):
        """Initialize OpenEnv client.

        Args:
            base_url: Base URL of the OpenEnv server (e.g., "http://localhost:8080")
            env_id: Optional environment ID to connect to
            timeout: Request timeout in seconds
            **env_kwargs: Additional kwargs passed to env on reset
        """
        self._base_url = base_url.rstrip("/")
        self._env_id = env_id
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._env_kwargs = env_kwargs
        self._session_id: str | None = None
        self._http_session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session (lazy initialization)."""
        if self._http_session is None or self._http_session.closed:
            self._http_session = aiohttp.ClientSession(timeout=self._timeout)
        return self._http_session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make HTTP request to the OpenEnv server."""
        url = f"{self._base_url}{endpoint}"
        session = await self._get_session()
        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def reset(self) -> StepResult:
        """Reset environment via API call."""
        payload = {"env_id": self._env_id, **self._env_kwargs}
        data = await self._request("POST", "/reset", json=payload)
        self._session_id = data["session_id"]
        return StepResult(
            observation=data["observation"], reward=data["reward"], done=data["done"], info=data.get("info", {})
        )

    async def step(self, action: dict[str, Any]) -> StepResult:
        """Execute action via API call."""
        payload = {"session_id": self._session_id, "action": action}
        data = await self._request("POST", "/step", json=payload)
        return StepResult(
            observation=data["observation"], reward=data["reward"], done=data["done"], info=data.get("info", {})
        )

    def state(self) -> dict[str, Any]:
        """Get state - not supported for remote envs in sync context."""
        return {"session_id": self._session_id}

    async def close(self):
        """Close the session on the remote server and HTTP client."""
        if self._session_id:
            await self._request("POST", "/close", json={"session_id": self._session_id})
            self._session_id = None
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
            self._http_session = None
