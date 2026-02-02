"""
OpenEnv client for connecting to remote OpenEnv servers.

OpenEnv is a standard for RL environment servers. The client sends
reset/step requests and receives observations, rewards, and tool schemas.
"""

import logging
from typing import Any

import aiohttp

from .base import RLEnvironment, ResetResult, StepResult, ToolCall, register_env

logger = logging.getLogger(__name__)


@register_env("openenv")
class OpenEnvClient(RLEnvironment):
    """
    Client for remote OpenEnv servers.

    OpenEnv servers expose a standard HTTP API:
    - POST /reset: Initialize episode, returns observation + tools
    - POST /step: Execute action, returns observation + reward + done

    Usage:
        # Start OpenEnv server externally
        # python -m openenv.servers.wordle --port 8765

        # Create client actor
        env = OpenEnvClient.remote(base_url="http://localhost:8765")

        # Run episode
        result = await env.reset.remote(task_id="game_001")
        step = await env.step.remote(ToolCall(name="guess", args={"word": "bread"}))
    """

    response_role = "tool"
    max_steps = 50

    def __init__(
        self,
        base_url: str,
        timeout: int = 30,
        headers: dict[str, str] | None = None,
    ):
        """
        Initialize OpenEnv client.

        Args:
            base_url: Base URL of the OpenEnv server (e.g., "http://localhost:8765")
            timeout: Request timeout in seconds (default: 30)
            headers: Optional HTTP headers to include in requests
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = headers or {}
        self._session: aiohttp.ClientSession | None = None
        self._current_tools: list[dict] = []
        self._step_count = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers=self._headers,
            )
        return self._session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make an HTTP request to the OpenEnv server."""
        session = await self._ensure_session()
        url = f"{self._base_url}{endpoint}"

        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def reset(self, task_id: str | None = None) -> ResetResult:
        """
        Reset the environment on the remote server.

        Args:
            task_id: Optional task identifier

        Returns:
            ResetResult with observation and tools from server
        """
        self._step_count = 0

        data = await self._request("POST", "/reset", json={"task_id": task_id})

        # Store tools for reference
        self._current_tools = data.get("tools", [])

        return ResetResult(
            observation=data.get("observation", ""),
            tools=self._current_tools,
            info=data.get("info", {}),
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        """
        Execute an action on the remote server.

        Args:
            tool_call: Parsed tool call to execute

        Returns:
            StepResult from server
        """
        self._step_count += 1

        data = await self._request(
            "POST",
            "/step",
            json={
                "tool_name": tool_call.name,
                "tool_args": tool_call.args,
                "tool_id": tool_call.id,
            },
        )

        return StepResult(
            observation=data.get("observation", ""),
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )

    def get_metrics(self) -> dict[str, float]:
        """Return client metrics."""
        return {
            "step_count": float(self._step_count),
        }

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


@register_env("openenv_repl")
class OpenEnvREPLClient(RLEnvironment):
    """
    Client for OpenEnv REPL environment servers.

    The REPL environment supports the RLM (Recursive Language Models) paradigm:
    - Sandboxed Python REPL execution
    - Optional llm_query() for recursive LLM calls
    - Built-in configurable rewards
    - Finalization via FINAL(answer)

    This is a specialized client that works with OpenEnv's repl_env.
    """

    response_role = "tool"
    max_steps = 30

    def __init__(
        self,
        base_url: str,
        context: str = "",
        task_prompt: str = "",
        timeout: int = 60,
        max_iterations: int = 30,
    ):
        """
        Initialize REPL client.

        Args:
            base_url: Base URL of the REPL OpenEnv server
            context: Context/document for the task
            task_prompt: Task description
            timeout: Request timeout in seconds
            max_iterations: Maximum REPL iterations
        """
        self._base_url = base_url.rstrip("/")
        self._context = context
        self._task_prompt = task_prompt
        self._timeout = timeout
        self._max_iterations = max_iterations
        self._session: aiohttp.ClientSession | None = None
        self._step_count = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Get or create the HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
            )
        return self._session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        """Make an HTTP request."""
        session = await self._ensure_session()
        url = f"{self._base_url}{endpoint}"

        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def reset(self, task_id: str | None = None) -> ResetResult:
        """
        Reset the REPL environment.

        Args:
            task_id: Optional task identifier (may override context/task_prompt)

        Returns:
            ResetResult with observation and execute tool
        """
        self._step_count = 0

        data = await self._request(
            "POST",
            "/reset",
            json={
                "task_id": task_id,
                "context": self._context,
                "task_prompt": self._task_prompt,
                "max_iterations": self._max_iterations,
            },
        )

        observation = data.get("observation", "")
        if isinstance(observation, dict):
            # Handle structured observation from REPL
            context_preview = observation.get("context_preview", "")
            observation = f"Context: {context_preview}...\nTask: {self._task_prompt}"

        return ResetResult(
            observation=observation,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "execute",
                        "description": "Execute Python code in the REPL. Use FINAL(answer) to submit your final answer.",
                        "parameters": {
                            "type": "object",
                            "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                            "required": ["code"],
                        },
                    },
                }
            ],
            info=data.get("info", {}),
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        """
        Execute code in the REPL.

        Args:
            tool_call: Should be execute(code="...")

        Returns:
            StepResult with REPL output and built-in reward
        """
        self._step_count += 1

        code = tool_call.args.get("code", "")

        data = await self._request(
            "POST",
            "/step",
            json={"code": code},
        )

        # Extract observation from REPL result
        observation = data.get("observation", "")
        if isinstance(observation, dict):
            result = observation.get("result", {})
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")
            observation = stdout if stdout else str(stderr) if stderr else "(no output)"

        return StepResult(
            observation=observation,
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )

    def get_metrics(self) -> dict[str, float]:
        """Return REPL metrics."""
        return {
            "step_count": float(self._step_count),
        }

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
