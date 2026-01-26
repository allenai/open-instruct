"""
AppWorld environment wrapper.

Wraps the existing appworld_env.py which uses AsyncServerPool for Docker container management.
"""

import atexit
from typing import Any

try:
    from appworld import AppWorld

    _APPWORLD_AVAILABLE = True
except ImportError:
    _APPWORLD_AVAILABLE = False

from open_instruct.environments.base import RLEnvironment, StepResult

# Global reference to AppWorld initializer for cleanup
_appworld_initializer = None


class AppWorldEnv(RLEnvironment):
    """Wraps the existing verifiers-based AppWorldEnv.

    Uses the AppWorld library for task execution and evaluation.
    Docker containers are managed via AppWorld.initializer().
    """

    def __init__(
        self, task_id: str, server_url: str, experiment_name: str = "default", ground_truth_mode: str = "partial"
    ):
        """Initialize AppWorld environment.

        Args:
            task_id: AppWorld task ID (e.g., "82e2fac_1")
            server_url: URL of the remote AppWorld server
            experiment_name: Name for experiment logging
            ground_truth_mode: "partial" or "full"
        """
        if not _APPWORLD_AVAILABLE:
            raise ImportError("appworld library required. Install with: pip install appworld")
        self._task_id = task_id
        self._server_url = server_url
        self._experiment_name = experiment_name
        self._ground_truth_mode = ground_truth_mode
        self._world = None

    def reset(self) -> StepResult:
        """Initialize AppWorld instance for this task."""
        self._world = AppWorld(
            task_id=self._task_id,
            experiment_name=self._experiment_name,
            remote_environment_url=self._server_url,
            load_ground_truth=True,
            ground_truth_mode=self._ground_truth_mode,
        )
        return StepResult(observation=f"Task {self._task_id} loaded. Ready for API calls.", reward=0.0, done=False)

    def step(self, action: dict[str, Any]) -> StepResult:
        """Execute API call on AppWorld.

        Args:
            action: Dict with "api_code" key containing the Python code to execute
        """
        api_code = action["api_code"]
        output = self._world.execute(api_code)

        # Check if task complete (supervisor__complete_task called)
        done = "supervisor__complete_task" in api_code

        # Get reward from evaluation if done
        reward = 0.0
        if done:
            reward = self._world.evaluate().pass_percentage * 0.01

        return StepResult(observation=str(output), reward=reward, done=done)

    def close(self):
        """Close the AppWorld instance."""
        self._world.close()


def _cleanup_appworld_servers():
    """Cleanup function to shut down AppWorld Docker containers."""
    global _appworld_initializer
    if _appworld_initializer is not None:
        _appworld_initializer.__exit__(None, None, None)
        _appworld_initializer = None


async def setup_appworld_servers(pool_size: int) -> list[str]:
    """Start AppWorld Docker containers, return server URLs.

    This is the setup_fn for EnvironmentPool. Registers cleanup via atexit.

    Args:
        pool_size: Number of servers to start

    Returns:
        List of server URLs
    """
    global _appworld_initializer
    if not _APPWORLD_AVAILABLE:
        raise ImportError("appworld library required. Install with: pip install appworld")

    config = {
        "experiment_name": "verification",
        "remote_environment_url": ["http://localhost:{port}"] * pool_size,
        "raise_on_failure": True,
        "ground_truth_mode": "partial",
    }
    _appworld_initializer = AppWorld.initializer(start_servers=True, **config)
    _appworld_initializer.__enter__()

    # Register cleanup to run at process exit
    atexit.register(_cleanup_appworld_servers)

    return [cfg["remote_environment_url"] for cfg in _appworld_initializer.configs]


def teardown_appworld_servers():
    """Explicitly shut down AppWorld Docker containers.

    Call this at end of training if you want to clean up before process exit.
    """
    _cleanup_appworld_servers()
