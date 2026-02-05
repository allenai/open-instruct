"""OpenEnv client for connecting to remote OpenEnv servers."""

import logging
from typing import Any

import aiohttp

from .base import RLEnvironment, StepResult, ToolCall, register_env

logger = logging.getLogger(__name__)


@register_env("openenv")
class OpenEnvClient(RLEnvironment):
    """Client for remote OpenEnv servers (HTTP API with /reset and /step endpoints)."""

    response_role = "tool"
    max_steps = 50

    def __init__(self, base_url: str, timeout: int = 30, headers: dict[str, str] | None = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = headers or {}
        self._session: aiohttp.ClientSession | None = None
        self._current_tools: list[dict] = []
        self._step_count = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout), headers=self._headers
            )
        return self._session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self._base_url}{endpoint}"

        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def reset(self, task_id: str | None = None) -> StepResult:
        """Reset the environment on the remote server."""
        self._step_count = 0
        data = await self._request("POST", "/reset", json={"task_id": task_id})
        self._current_tools = data.get("tools", [])

        return StepResult(
            observation=data.get("observation", ""), tools=self._current_tools, info=data.get("info", {})
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        """Execute an action on the remote server."""
        self._step_count += 1
        data = await self._request(
            "POST", "/step", json={"tool_name": tool_call.name, "tool_args": tool_call.args, "tool_id": tool_call.id}
        )

        return StepResult(
            observation=data.get("observation", ""),
            reward=float(data.get("reward", 0.0)),
            done=bool(data.get("done", False)),
            info=data.get("info", {}),
        )

    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count)}

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


@register_env("openenv_text")
class OpenEnvTextClient(RLEnvironment):
    """Client for text-based OpenEnv servers (e.g., textarena).

    Unlike OpenEnvClient which expects tool calls, this client sends the raw
    model output as a text message. Use this for environments that expect
    free-form text responses rather than structured tool calls.
    """

    response_role = "assistant"  # Text mode: model output goes to assistant role
    observation_role = "user"  # Environment feedback wrapped in user role
    max_steps = 50

    def __init__(self, base_url: str, timeout: int = 30, headers: dict[str, str] | None = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._headers = headers or {}
        self._session: aiohttp.ClientSession | None = None
        self._step_count = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout), headers=self._headers
            )
        return self._session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self._base_url}{endpoint}"

        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def reset(self, task_id: str | None = None) -> StepResult:
        """Reset the environment on the remote server.

        Note: We return an empty observation because text-based environments
        like TextArena return verbose instructions that confuse the model.
        The dataset prompt should already contain the necessary instructions.
        """
        self._step_count = 0
        await self._request("POST", "/reset", json={"task_id": task_id})

        # Return empty observation - the dataset prompt already has instructions
        # and TextArena's verbose instructions confuse small models
        return StepResult(observation="", tools=[], info={})

    async def step(self, tool_call: ToolCall) -> StepResult:
        """Execute a text action on the remote server.

        For text-based environments, the tool_call.args["message"] contains
        the raw text to send. This is a workaround since the base interface
        expects ToolCall objects.
        """
        self._step_count += 1
        # Get the message from args, or use the tool name as fallback
        message = tool_call.args.get("message", tool_call.name)
        data = await self._request("POST", "/step", json={"action": {"message": message}})

        # Extract observation - handle both dict and string formats
        obs_data = data.get("observation", "")
        info = data.get("info", {})

        if isinstance(obs_data, dict):
            # TextArena returns observation as dict with messages and info
            messages = obs_data.get("messages", [])
            # Extract info from observation dict (TextArena puts reward_signals here)
            if "info" in obs_data:
                info = obs_data["info"]
            if messages:
                # Get the last message content
                content = messages[-1].get("content", str(obs_data))
                # Extract just the feedback portion for cleaner context
                # TextArena includes full history, but we only need the latest feedback
                observation = self._extract_feedback(content)
            else:
                observation = obs_data.get("prompt", str(obs_data))
        else:
            observation = obs_data

        # Compute reward from reward_signals if available (TextArena provides shaped rewards)
        # Otherwise use the main reward field
        reward = float(data.get("reward", 0.0))
        if "reward_signals" in info:
            signals = info["reward_signals"]
            # Sum all reward signals - they're typically normalized to [0, 1]
            # wordle.greens: 0.2 per green (max 1.0), wordle.yellows: 0.2 per yellow (max 1.0)
            # wordle.correct: 1.0 if solved, wordle.repetitions: bonus for diverse letters
            reward = sum(signals.values())

        return StepResult(observation=observation, reward=reward, done=bool(data.get("done", False)), info=info)

    def _extract_feedback(self, content: str) -> str:
        """Extract just the feedback portion from TextArena's verbose response.

        TextArena returns full game history, but we only want the latest feedback
        to keep context clean and avoid confusing the model.
        """
        # Look for the feedback section which starts with "Feedback:" or after "[GAME] You submitted"
        lines = content.split("\n")

        # Find the last "[GAME] You submitted" line and extract feedback after it
        feedback_start = -1
        for i, line in enumerate(lines):
            if "[GAME] You submitted" in line or "Feedback:" in line:
                feedback_start = i

        if feedback_start >= 0:
            # Return from feedback start to end, but keep it concise
            feedback_lines = lines[feedback_start:]
            return "\n".join(feedback_lines).strip()

        # For invalid moves or other responses, look for error messages
        for line in lines:
            if "invalid move" in line.lower() or "error" in line.lower():
                return line.strip()

        # Fallback: return last few lines if no pattern matched
        return "\n".join(lines[-4:]).strip() if lines else content

    def get_metrics(self) -> dict[str, float]:
        return {"step_count": float(self._step_count)}

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


@register_env("openenv_repl")
class OpenEnvREPLClient(RLEnvironment):
    """Client for OpenEnv REPL environment servers with sandboxed Python execution."""

    response_role = "tool"
    max_steps = 30

    def __init__(
        self, base_url: str, context: str = "", task_prompt: str = "", timeout: int = 60, max_iterations: int = 30
    ):
        self._base_url = base_url.rstrip("/")
        self._context = context
        self._task_prompt = task_prompt
        self._timeout = timeout
        self._max_iterations = max_iterations
        self._session: aiohttp.ClientSession | None = None
        self._step_count = 0

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self._timeout))
        return self._session

    async def _request(self, method: str, endpoint: str, **kwargs) -> dict[str, Any]:
        session = await self._ensure_session()
        url = f"{self._base_url}{endpoint}"

        async with session.request(method, url, **kwargs) as response:
            response.raise_for_status()
            return await response.json()

    async def reset(self, task_id: str | None = None) -> StepResult:
        """Reset the REPL environment."""
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
            context_preview = observation.get("context_preview", "")
            observation = f"Context: {context_preview}...\nTask: {self._task_prompt}"

        return StepResult(
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
        """Execute code in the REPL."""
        self._step_count += 1
        code = tool_call.args.get("code", "")
        data = await self._request("POST", "/step", json={"code": code})

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
        return {"step_count": float(self._step_count)}

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
