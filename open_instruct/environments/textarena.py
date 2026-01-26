"""
Adapter for TextArena environments via OpenEnv.

TextArena provides games like Wordle through the OpenEnv standard.
Run server with: TEXTARENA_ENV_ID=Wordle-v0 uvicorn textarena_env.server.app:app
"""

from typing import Any

try:
    from textarena_env import TextArenaAction, TextArenaEnv as TextArenaClient

    _TEXTARENA_AVAILABLE = True
except ImportError:
    _TEXTARENA_AVAILABLE = False

from open_instruct.environments.base import RLEnvironment, StepResult


class TextArenaEnv(RLEnvironment):
    """Adapter for TextArena environments accessed via HTTP server.

    Wraps the TextArena client to provide the RLEnvironment interface.
    """

    def __init__(self, base_url: str, **kwargs: Any):
        """Initialize TextArena client.

        Args:
            base_url: URL of the TextArena server (e.g., "http://localhost:8000")
            **kwargs: Additional kwargs (unused, for compatibility)
        """
        if not _TEXTARENA_AVAILABLE:
            raise ImportError("textarena-env required. Install with: uv pip install textarena-env")
        self._base_url = base_url
        self._client = TextArenaClient(base_url=base_url)
        self._last_observation: str = ""

    def reset(self) -> StepResult:
        """Reset environment and get initial observation."""
        result = self._client.reset()
        # Extract the game prompt/instructions
        self._last_observation = result.observation.prompt
        return StepResult(
            observation=self._last_observation,
            reward=result.reward,
            done=result.done,
            info={"messages": [], "current_player_id": result.observation.current_player_id},
        )

    async def step(self, message: str) -> StepResult:
        """Execute action (submit a guess/message to the game).

        Args:
            message: The message to send (e.g., "[crane]" for Wordle guess)
        """
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
            info={
                "messages": messages_content,
                "current_player_id": result.observation.current_player_id,
            },
        )

    def close(self):
        """Close the TextArena client."""
        if self._client:
            self._client.close()
