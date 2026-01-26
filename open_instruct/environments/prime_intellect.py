"""
Adapter for Prime Intellect verifiers-based environments.

Prime Intellect environments use the verifiers library spec. Install with:
    uv pip install .[prime-intellect]
    prime env install will/wordle
    prime env install will/wiki-search
"""

import signal
from contextlib import contextmanager
from typing import Any

try:
    from verifiers import load_environment

    _VERIFIERS_AVAILABLE = True
except ImportError:
    _VERIFIERS_AVAILABLE = False

from open_instruct.environments.base import RLEnvironment, StepResult


@contextmanager
def _patch_signal_for_non_main_thread():
    """Patch signal.signal to be a no-op when called from non-main thread.

    The verifiers library uses signal handlers for timeouts, but signal.signal()
    can only be called from the main thread. When running in Ray workers, we're
    in a non-main thread, so we need to disable signal handling.

    This is safe because:
    1. We handle timeouts at a higher level (vLLM generation timeouts)
    2. The environment step operations are async and have their own timeout mechanisms
    """
    original_signal = signal.signal

    def _noop_signal(signum, handler):
        # Silently ignore signal registration in non-main threads
        return signal.SIG_DFL

    signal.signal = _noop_signal
    try:
        yield
    finally:
        signal.signal = original_signal


class PrimeIntellectEnv(RLEnvironment):
    """Adapts a verifiers MultiTurnEnv to RLEnvironment interface.

    Wraps verifiers' env_response()/is_completed() into step()/reset().
    """

    def __init__(self, env_name: str, **env_kwargs: Any):
        """Initialize Prime Intellect environment.

        Args:
            env_name: Name of the PI env (e.g., "will/wordle", "will/wiki-search")
            **env_kwargs: Additional kwargs passed to the verifiers env
        """
        if not _VERIFIERS_AVAILABLE:
            raise ImportError("verifiers library required. Install with: uv pip install .[prime-intellect]")
        self._env_name = env_name
        self._env_kwargs = env_kwargs
        self._vf_env = None
        self._state: dict[str, Any] = {}
        self._messages: list[dict[str, str]] = []

    def reset(self) -> StepResult:
        """Initialize episode, load verifiers env."""
        with _patch_signal_for_non_main_thread():
            self._vf_env = load_environment(self._env_name, **self._env_kwargs)
        self._state = {"turn": 0, "prompt": "", "completion": []}
        self._messages = []
        return StepResult("Environment ready.", reward=0.0, done=False)

    async def step(self, action: dict[str, Any]) -> StepResult:
        """Execute action on env.

        Args:
            action: Dict with "content" key containing the model's message/tool call
        """
        self._messages.append({"role": "assistant", "content": action["content"]})

        # Get env response (async in verifiers)
        env_response, self._state = await self._vf_env.env_response(self._messages, self._state)
        self._messages.extend(env_response)

        # Check completion
        done = await self._vf_env.is_completed(self._messages, self._state)

        # Get reward from rubric if done
        reward = 0.0
        if done:
            scores = await self._vf_env.rubric.score_rollout(
                self._state["prompt"], self._messages, self._state["answer"], self._state
            )
            reward = scores["reward"]

        observation = env_response[-1]["content"]
        return StepResult(observation, reward=reward, done=done)

    def close(self):
        """Cleanup - verifiers envs don't need explicit cleanup."""
        pass
