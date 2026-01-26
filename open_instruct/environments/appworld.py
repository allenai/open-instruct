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


class AppWorldServerManager:
    """Manages AppWorld Docker container lifecycle.

    Encapsulates server management for better modularity and testability.
    """

    _instance: "AppWorldServerManager | None" = None

    def __init__(self, pool_size: int):
        """Initialize server manager.

        Args:
            pool_size: Number of Docker containers to start.
        """
        if not _APPWORLD_AVAILABLE:
            raise ImportError("appworld library required. Install with: pip install appworld")
        self.pool_size = pool_size
        self._initializer = None
        self._server_urls: list[str] = []

    async def start(self) -> list[str]:
        """Start AppWorld Docker containers.

        Returns:
            List of server URLs.
        """
        config = {
            "experiment_name": "verification",
            "remote_environment_url": ["http://localhost:{port}"] * self.pool_size,
            "raise_on_failure": True,
            "ground_truth_mode": "partial",
        }
        self._initializer = AppWorld.initializer(start_servers=True, **config)
        self._initializer.__enter__()
        self._server_urls = [cfg["remote_environment_url"] for cfg in self._initializer.configs]

        # Register cleanup via atexit as a fallback
        atexit.register(self.stop)

        return self._server_urls

    def stop(self):
        """Stop AppWorld Docker containers."""
        if self._initializer is not None:
            self._initializer.__exit__(None, None, None)
            self._initializer = None
            self._server_urls = []

    @property
    def server_urls(self) -> list[str]:
        """Get list of server URLs."""
        return self._server_urls


# Module-level manager instance for backward compatibility
_server_manager: AppWorldServerManager | None = None


async def setup_appworld_servers(pool_size: int) -> list[str]:
    """Start AppWorld Docker containers, return server URLs.

    This is the setup_fn for EnvironmentPool.

    Args:
        pool_size: Number of servers to start

    Returns:
        List of server URLs
    """
    global _server_manager
    _server_manager = AppWorldServerManager(pool_size)
    return await _server_manager.start()


def teardown_appworld_servers():
    """Explicitly shut down AppWorld Docker containers.

    Call this at end of training if you want to clean up before process exit.
    """
    global _server_manager
    if _server_manager is not None:
        _server_manager.stop()
        _server_manager = None
