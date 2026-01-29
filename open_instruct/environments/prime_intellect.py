"""
Adapter for Prime Intellect verifiers-based environments.

Prime Intellect environments use the verifiers library spec. Install with:
    uv pip install .[prime-intellect]
    prime env install will/wordle
    prime env install will/wiki-search
"""

import asyncio
import random
import signal
import time
from contextlib import contextmanager
from typing import Any, ClassVar

try:
    from verifiers import load_environment

    _VERIFIERS_AVAILABLE = True
except ImportError:
    _VERIFIERS_AVAILABLE = False

from open_instruct import logger_utils
from open_instruct.environments.base import RLEnvironment, StepResult

logger = logger_utils.setup_logger(__name__)


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
    The base environment is loaded once per env_name and shared across instances.
    Each reset() samples a new problem from verifiers' internal dataset.
    """

    # Cache: env_name -> loaded base environment (shared across all instances)
    _base_envs: ClassVar[dict[str, Any]] = {}

    def __init__(self, env_name: str, **env_kwargs: Any):
        """Initialize Prime Intellect environment.

        Args:
            env_name: Name of the PI env (e.g., "will/wordle", "will/wiki-search")
            **env_kwargs: Additional kwargs (currently unused, env samples from its dataset)
        """
        if not _VERIFIERS_AVAILABLE:
            raise ImportError("verifiers library required. Install with: uv pip install .[prime-intellect]")
        self._env_name = env_name
        self._state: dict[str, Any] = {}
        self._messages: list[dict[str, str]] = []

        # Load base env ONCE per env_name (shared across all instances)
        if env_name not in PrimeIntellectEnv._base_envs:
            with _patch_signal_for_non_main_thread():
                PrimeIntellectEnv._base_envs[env_name] = load_environment(env_name)
        self._vf_env = PrimeIntellectEnv._base_envs[env_name]

    async def reset(self) -> StepResult:
        """Reset episode state by sampling a new problem from verifiers' dataset."""
        logger.info(f"[PrimeIntellectEnv] reset() called for {self._env_name}")
        # Sample a problem from verifiers' internal dataset
        dataset = self._vf_env.get_dataset()
        sample_idx = random.randint(0, len(dataset) - 1)
        sample = dataset[sample_idx]
        logger.info(f"[PrimeIntellectEnv] sampled index {sample_idx}, setting up state...")

        # Initialize state from sample, add fields required by verifiers
        self._state = {
            **sample,  # Unpack all fields from the dataset sample
            "turn": 0,
            "completion": [],
            "trajectory": [],
            # Timing dict required by verifiers for scoring
            "timing": {"start_time": time.time(), "scoring_ms": 0.0, "total_ms": 0.0, "generation_ms": 0.0},
        }
        # Setup verifiers state (e.g., initializes ta_env for TextArena environments)
        logger.info("[PrimeIntellectEnv] calling setup_state...")
        await self._vf_env.setup_state(self._state)
        logger.info("[PrimeIntellectEnv] setup_state complete")
        self._messages = []
        return StepResult("Environment ready.", reward=0.0, done=False)

    async def step(self, action: dict[str, Any]) -> StepResult:
        """Execute action on env.

        Args:
            action: Dict with tool call arguments (e.g., {'guess': 'BREAD'} for wordle)
                    or "content" key containing the model's message/tool call
        """
        logger.info(f"[PrimeIntellectEnv] step() called with action: {str(action)[:100]}...")
        # Handle action formats:
        # 1. {'content': '<guess>BREAD</guess>'} - raw XML content from model (fallback)
        # 2. {'guess': 'BREAD'} - structured tool call arguments (converted to XML)

        # Translate parameter names for compatibility (e.g., "word" -> "guess" for Wordle)
        if "word" in action and "guess" not in action:
            action = {**action, "guess": action.pop("word")}

        content = action["content"] if "content" in action else "".join(f"<{k}>{v}</{k}>" for k, v in action.items())
        assistant_msg = {"role": "assistant", "content": content}
        self._messages.append(assistant_msg)
        self._state["completion"].append(assistant_msg)

        # Get env response (async in verifiers) - returns Messages, state is mutated in place
        logger.info("[PrimeIntellectEnv] calling env_response...")
        env_response = await self._vf_env.env_response(self._messages, self._state)
        logger.info(f"[PrimeIntellectEnv] env_response returned {len(env_response)} messages")
        self._messages.extend(env_response)
        self._state["completion"].extend(env_response)

        # Check completion using has_final_env_response (more reliable than is_completed)
        done = await self._vf_env.has_final_env_response(self._state)
        logger.info(f"[PrimeIntellectEnv] has_final_env_response returned done={done}")

        # Get reward from rubric if done
        reward = 0.0
        if done:
            logger.info("[PrimeIntellectEnv] scoring rollout...")
            score_sem = asyncio.Semaphore(1)
            await self._vf_env.rubric.score_rollout(self._state, score_sem)
            reward = self._state.get("reward", 0.0)
            logger.info(f"[PrimeIntellectEnv] score_rollout complete, reward={reward}")

        observation = env_response[-1]["content"]
        logger.info(f"[PrimeIntellectEnv] step() returning, obs: {observation[:100] if observation else ''}")
        return StepResult(observation, reward=reward, done=done)

    def close(self):
        """Cleanup - verifiers envs don't need explicit cleanup."""
        pass
