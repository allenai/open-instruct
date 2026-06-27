"""AppWorld code-as-action RL environment (container + HTTP).

Wraps the AppWorld benchmark (https://appworld.dev, ACL'24 best resource paper)
as an open-instruct RL environment. The policy solves a task by writing Python
that calls ``apis.<app>.<endpoint>(...)`` against a stateful, simulated world of
apps (Amazon, Spotify, Venmo, phone, file system, ...) on behalf of a supervisor,
and submits by calling ``apis.supervisor.complete_task()``. Reward is AppWorld's
own programmatic evaluation: the fraction of unit tests passed (dense) or whether
*all* tests pass (binary Task Goal Completion).

Why a container + HTTP (not in-process)
---------------------------------------
AppWorld pins ``pydantic<2`` while open-instruct/openenv require ``pydantic>=2``;
they cannot share a Python process. AppWorld ships an HTTP *environment server*
for exactly this case, so we mirror the Terminal RL (swerl) podman/docker workflow:

* One AppWorld container per rollout (the server holds a single world at a time,
  matching swerl's one-container-per-rollout ratio), reused across resets.
* The container runs ``appworld serve environment``; this env is a thin HTTP client
  (``/initialize``, ``/execute``, ``/task_completed``, ``/evaluate``, ``/close``) and
  never imports ``appworld`` — so the trainer process stays pydantic-2-clean.
* Unlike swerl (stateless ``docker exec bash`` per turn), AppWorld is a stateful
  Python REPL: the agent's variables and the world object must persist in one
  long-lived process, hence the in-container server + HTTP rather than exec-per-turn.

The container image (``ghcr.io/stonybrooknlp/appworld:latest``) bundles AppWorld;
the host provides the task data via a bind mount (``data_root``). Model-generated
code runs inside the container, isolating it from the trainer.
"""

import contextlib
import socket
import time
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import urlparse

import requests
from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .tools.utils import coerce_args

logger = logger_utils.setup_logger(__name__)

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute_python",
        "description": (
            "Execute a snippet of Python code in a persistent IPython session. "
            "The namespace already contains `apis` (the AppWorld app APIs) and persists "
            "across calls, so variables defined in one call are available in the next. "
            "Use `print(...)` to inspect values; only printed output is returned. "
            "Discover APIs with `print(apis.api_docs.show_app_descriptions())`, "
            "`print(apis.api_docs.show_api_descriptions(app_name='...'))`, and "
            "`print(apis.api_docs.show_api_doc(app_name='...', api_name='...'))`. "
            "Call `apis.supervisor.complete_task()` when the task is finished."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to execute."}},
            "required": ["code"],
        },
    },
}

# Canonical code-as-action system prompt, adapted from AppWorld's reference REACT
# agent for the single-tool ``execute_python`` interface. Kept here (not in the data
# converter) so the env and the dataset builder share one source of truth.
SYSTEM_PROMPT = """\
You are a coding agent who interacts with a digital world of apps by writing and \
executing Python code. You complete a task on behalf of your supervisor by calling \
the apps' APIs through the `apis` object.

How you operate:
- At each step you write a short snippet of Python code. It runs in a persistent \
IPython session; only what you `print(...)` is shown back to you, and your variables \
persist across steps.
- The `apis` object exposes one attribute per app. Read the docs before calling \
anything you are unsure about:
  - `print(apis.api_docs.show_app_descriptions())` lists the apps.
  - `print(apis.api_docs.show_api_descriptions(app_name='amazon'))` lists an app's APIs.
  - `print(apis.api_docs.show_api_doc(app_name='amazon', api_name='search_products'))` \
shows one API's arguments and response schema.
- Most apps require an access token. Get the supervisor's per-app passwords with \
`print(apis.supervisor.show_account_passwords())`, then log in, e.g. \
`apis.<app>.login(username=<supervisor email or phone>, password=<that app's password>)`, \
and pass the returned access token to subsequent calls.
- Work incrementally: inspect results, handle pagination, and verify before acting. \
Write robust code (loops, error handling) rather than guessing.
- When, and only when, the task is fully done, call `apis.supervisor.complete_task()` \
(pass `answer=...` if the task asks for an answer). Do not call it prematurely.

Your supervisor's details:
{supervisor_block}
"""

USER_PROMPT = "Task:\n{instruction}"

# Observation truncation (head/tail) for long execution output.
_OBS_MAX_CHARS = 8000
_OBS_HEAD_CHARS = 4000
_OBS_TAIL_CHARS = 3000


def _supervisor_dict(supervisor: Any) -> dict[str, str]:
    """Extract the supervisor fields an agent needs to log in, robustly."""
    fields = ("first_name", "last_name", "email", "phone_number", "password")
    out: dict[str, str] = {}
    for key in fields:
        value = getattr(supervisor, key, None)
        if value is None and isinstance(supervisor, dict):
            value = supervisor.get(key)
        if value is not None:
            out[key] = str(value)
    return out


def format_supervisor_block(supervisor: Any) -> str:
    info = _supervisor_dict(supervisor)
    label = {
        "first_name": "First name",
        "last_name": "Last name",
        "email": "Email",
        "phone_number": "Phone number",
        "password": "Password",
    }
    lines = [f"- {label.get(k, k)}: {v}" for k, v in info.items()]
    return "\n".join(lines) if lines else "- (no supervisor details provided)"


def build_prompt_messages(instruction: str, supervisor: Any) -> list[dict[str, str]]:
    """Build the chat messages for an AppWorld task.

    The env's :meth:`reset` observation is discarded by the rollout pipeline, so
    the task instruction and supervisor credentials must live in the dataset
    ``messages``. The data converter calls this to bake the prompt into each row.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(supervisor_block=format_supervisor_block(supervisor))},
        {"role": "user", "content": USER_PROMPT.format(instruction=instruction)},
    ]


def truncate_observation(output: str) -> str:
    if len(output) <= _OBS_MAX_CHARS:
        return output
    elided = len(output) - _OBS_HEAD_CHARS - _OBS_TAIL_CHARS
    return f"{output[:_OBS_HEAD_CHARS]}\n---- {elided} chars elided ----\n{output[-_OBS_TAIL_CHARS:]}"


def reward_from_evaluation(evaluation: dict, dense: bool) -> tuple[float, dict[str, float]]:
    """Map the AppWorld ``/evaluate`` payload to a scalar reward + metrics.

    The server returns ``TestTracker.to_dict()``: ``passes`` (list), ``failures``
    (list), ``num_tests`` (int), ``success`` (bool). Dense reward is the pass
    fraction; binary reward is all-pass (Task Goal Completion).
    """
    num_tests = int(evaluation.get("num_tests", 0) or 0)
    pass_count = len(evaluation.get("passes", []) or [])
    success = bool(evaluation.get("success", num_tests > 0 and pass_count == num_tests))
    dense_reward = pass_count / num_tests if num_tests > 0 else 0.0
    reward = dense_reward if dense else (1.0 if success else 0.0)
    metrics = {"num_tests": float(num_tests), "pass_count": float(pass_count), "success": float(success)}
    return max(0.0, min(1.0, reward)), metrics


class AppWorldEnv(RLEnvironment):
    """AppWorld code-as-action environment: a per-rollout container + HTTP client."""

    config_name = "appworld"
    response_role = "tool"
    _tool_definitions = (EXECUTE_TOOL,)

    def __init__(
        self,
        image: str = "ghcr.io/stonybrooknlp/appworld:latest",
        data_root: str = "",
        in_container_root: str = "/run",
        server_port: int = 8000,
        max_interactions: int = 50,
        timeout_seconds: int = 120,
        startup_timeout: int = 180,
        dense_reward: bool = True,
        mem_limit: str = "4g",
        experiment_prefix: str = "appworld_rl",
        backend: str = "docker",
        publish_port: bool = False,
        docker_host: str | None = None,
        **kwargs: Any,
    ):
        # Tolerate generic config kwargs injected by the pool (e.g. call_name, penalty).
        kwargs.pop("call_name", None)
        kwargs.pop("penalty", None)
        # ``backend`` exists so EnvironmentPool engages SWERL_PODMAN_DOCKER_HOSTS
        # rotation (it keys on backend == "docker") and injects a docker_host into
        # reset kwargs on beaker; we always use the docker SDK regardless of value.
        self._backend = backend
        self._image = image
        self._data_root = data_root
        self._in_container_root = in_container_root.rstrip("/")
        self._server_port = server_port
        self._max_interactions = max_interactions
        self._timeout_seconds = timeout_seconds
        self._startup_timeout = startup_timeout
        self._dense_reward = dense_reward
        self._mem_limit = mem_limit
        self._experiment_prefix = experiment_prefix
        self._publish_port = publish_port
        self._docker_host = docker_host

        self._client: Any = None  # docker client (lazily imported)
        self._container: Any = None
        self._port = server_port  # per-container server port (reassigned in _start_container)
        self._base_url: str | None = None
        self._task_id: str | None = None
        self._experiment_name: str | None = None
        self._step_count = 0
        self._completed = False
        self._last_reward = 0.0
        self._last_eval: dict[str, float] = {}
        self._max_steps: int | None = None

    async def setup(self) -> None:
        """Create the docker client. AppWorld is never imported in this process."""
        if self._client is not None:
            return
        import docker  # noqa: PLC0415 -- optional dep; keep out of module import path

        self._client = (
            docker.DockerClient(base_url=self._docker_host) if self._docker_host else docker.from_env()
        )

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return list(cls._tool_definitions)

    # ---- container lifecycle -------------------------------------------------

    def _resolve_host(self) -> str:
        """Host to reach published container ports on (bridge IP for remote podman)."""
        if self._docker_host:
            parsed = urlparse(self._docker_host)
            if parsed.hostname:
                return parsed.hostname
        return "127.0.0.1"

    @staticmethod
    def _pick_free_port() -> int:
        """Grab a currently-free TCP port in this process's network namespace.

        On Beaker the podman services run containers on the *host* network (shared with this
        trainer netns), so every AppWorld server binds 127.0.0.1 here and concurrent
        containers must use distinct ports — hence one free port per container. A bind race
        (container takes the port between close() and start) just fails startup and the reset
        retries with a new port.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def _start_container(self) -> None:
        # Bind-mount the host data when data_root is set (Beaker/weka nodes); when it
        # is empty, the image is expected to already contain the data at
        # {in_container_root}/data (a data-baked image, e.g. for envs without a shared FS).
        volumes = {}
        if self._data_root:
            volumes = {
                f"{self._data_root}/data": {"bind": f"{self._in_container_root}/data", "mode": "ro"},
                f"{self._data_root}/experiments/outputs": {
                    "bind": f"{self._in_container_root}/experiments/outputs",
                    "mode": "rw",
                },
            }
        # Unique port per container: Beaker's podman defaults containers onto the *host*
        # network (no bridge IP, publishing is a no-op), so the server is reached at
        # 127.0.0.1:<port> and concurrent containers cannot share one port.
        self._port = self._pick_free_port()
        container = self._client.containers.run(
            self._image,
            command=["environment", "--port", str(self._port), "--no-show-usage"],
            environment={"APPWORLD_ROOT": self._in_container_root},
            volumes=volumes,
            mem_limit=self._mem_limit,
            memswap_limit=self._mem_limit,
            detach=True,
            # auto_remove is intentionally OFF: an instantly-exiting container would be
            # deleted before we could read its logs/exit code. We remove it ourselves in
            # teardown / on startup failure instead.
            auto_remove=False,
            labels={"appworld_rl": "1"},
        )
        self._container = container
        try:
            self._base_url = self._wait_for_server(container)
        except Exception:
            self._remove_container(container)
            self._container = None
            raise

    @staticmethod
    def _remove_container(container: Any) -> None:
        with contextlib.suppress(Exception):
            container.remove(force=True)

    def _container_exit_detail(self, container: Any) -> str:
        """Exit code + log tail of a container that died — the actual failure reason."""
        try:
            container.reload()
            state = container.attrs.get("State", {}) or {}
            logs = container.logs(tail=60).decode("utf-8", "replace")
            return f"exit_code={state.get('ExitCode')} error={state.get('Error')!r}\n--- container logs (tail) ---\n{logs}"
        except Exception as e:
            return f"(could not read container logs: {e})"

    def _candidate_base_urls(self, container: Any) -> list[str]:
        """Candidate base URLs to reach the in-container server, most-portable first.

        Reachability differs by setup, so we probe and use whichever answers:
        - ``127.0.0.1:<port>`` — Beaker podman defaults containers onto the host network
          (shared with this netns), so the server is here. Primary path.
        - container bridge IP — local docker with the default bridge network.
        - published host port — only if port publishing was enabled and honored.
        """
        candidates: list[str] = [f"http://127.0.0.1:{self._port}"]
        container.reload()
        net = container.attrs.get("NetworkSettings", {}) or {}
        ip = net.get("IPAddress") or next(
            (n.get("IPAddress") for n in (net.get("Networks") or {}).values() if n.get("IPAddress")), None
        )
        if ip:
            candidates.append(f"http://{ip}:{self._port}")
        mapping = (net.get("Ports") or {}).get(f"{self._port}/tcp")
        if mapping:
            candidates.append(f"http://{self._resolve_host()}:{mapping[0]['HostPort']}")
        return candidates

    def _wait_for_server(self, container: Any) -> str:
        deadline = time.monotonic() + self._startup_timeout
        candidates: list[str] = []
        while time.monotonic() < deadline:
            try:
                container.reload()
            except Exception as e:
                raise RuntimeError(
                    f"AppWorld container vanished during startup (cannot inspect): {e}. "
                    "If this is podman with auto-remove, the container exited instantly."
                ) from e
            if container.status not in ("created", "running"):
                raise RuntimeError(
                    f"AppWorld container exited early (status={container.status}). "
                    f"{self._container_exit_detail(container)}"
                )
            candidates = self._candidate_base_urls(container)
            for base_url in candidates:
                with contextlib.suppress(requests.RequestException):
                    if requests.get(f"{base_url}/", timeout=5).status_code == 200:
                        logger.info(f"[{self._task_id}] AppWorld server reachable at {base_url}")
                        return base_url
            time.sleep(1.0)
        raise RuntimeError(
            f"AppWorld server did not become ready within {self._startup_timeout}s "
            f"(tried {candidates}). {self._container_exit_detail(container)}"
        )

    def _http(self, method: str, **body: Any) -> Any:
        if self._base_url is None:
            raise RuntimeError("AppWorld container not started.")
        response = requests.post(f"{self._base_url}/{method}", json=body, timeout=self._timeout_seconds + 60)
        if response.status_code != 200:
            raise RuntimeError(f"AppWorld /{method} failed ({response.status_code}): {response.text[:500]}")
        return response.json()["output"]

    # ---- env API -------------------------------------------------------------

    async def reset(self, task_id: str | None = None, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        if self._client is None:
            await self.setup()
        if task_id is None:
            raise ValueError("AppWorldEnv.reset requires a task_id (set it via env_config).")
        # Allow the pool to rotate docker hosts on retry (mirrors swerl).
        if kwargs.get("docker_host") and kwargs["docker_host"] != self._docker_host:
            self._docker_host = kwargs["docker_host"]
            self._teardown_container()
            await self.setup()
        if self._container is None or self._base_url is None:
            self._start_container()

        self._step_count = 0
        self._completed = False
        self._last_reward = 0.0
        self._last_eval = {}
        self._task_id = task_id
        self._max_steps = kwargs.get("max_steps", self._max_interactions)
        episode = kwargs.get("episode_id") or kwargs.get("instance_id") or uuid.uuid4().hex
        self._experiment_name = f"{self._experiment_prefix}_{task_id}_{episode}"

        # Re-initialize the container's single world to this task.
        self._http(
            "initialize",
            task_id=task_id,
            experiment_name=self._experiment_name,
            max_interactions=self._max_interactions,
            timeout_seconds=self._timeout_seconds,
            load_ground_truth=True,  # required for /evaluate
            raise_on_unsafe_syntax=True,
            null_patch_unsafe_execution=True,
            raise_on_failure=False,
        )
        # The reset observation is discarded by the rollout pipeline; the prompt
        # comes from the dataset messages (see build_prompt_messages).
        return StepResult(result="", metadata={"task_id": task_id}), list(self._tool_definitions)

    async def step(self, call: EnvCall) -> StepResult:
        if self._base_url is None or self._task_id is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        self._step_count += 1

        if call.name != "execute_python":
            return StepResult(
                result=f"Unknown tool '{call.name}'. The only available tool is `execute_python`."
            )

        args = coerce_args(EXECUTE_TOOL["function"]["parameters"], call.args)
        code = args.get("code", "")
        if not code or not code.strip():
            return StepResult(result="Error: the `code` parameter is required and must be non-empty.")

        output = self._http("execute", task_id=self._task_id, code=code)
        observation = truncate_observation(output) if output else "(no output)"

        self._completed = bool(self._http("task_completed", task_id=self._task_id))
        at_step_limit = self._max_steps is not None and self._step_count >= self._max_steps
        if self._completed or at_step_limit:
            reward = self._compute_reward()
            summary = (
                f"\n\n[Task {'completed' if self._completed else 'truncated at step limit'}. "
                f"Reward {reward:.3f} "
                f"({self._last_eval.get('pass_count', 0):.0f}/{self._last_eval.get('num_tests', 0):.0f} tests passed).]"
            )
            return StepResult(
                result=observation + summary,
                reward=reward,
                done=True,
                metadata={"task_id": self._task_id, "completed": self._completed, **self._last_eval},
            )
        return StepResult(result=observation, metadata={"task_id": self._task_id, "completed": False})

    def _compute_reward(self) -> float:
        try:
            evaluation = self._http("evaluate", task_id=self._task_id, suppress_errors=True, report=False)
            reward, self._last_eval = reward_from_evaluation(evaluation, self._dense_reward)
        except Exception as e:
            logger.warning(f"[{self._task_id}] AppWorld evaluation failed: {e}")
            self._last_eval = {"num_tests": 0.0, "pass_count": 0.0, "success": 0.0}
            reward = 0.0
        self._last_reward = reward
        return reward

    def get_metrics(self) -> dict[str, float]:
        return {
            "step_count": float(self._step_count),
            "task_completed": float(self._completed),
            "reward": float(self._last_reward),
            **self._last_eval,
        }

    def state(self) -> State:
        return State(episode_id=self._task_id, step_count=self._step_count)

    def _teardown_container(self) -> None:
        if self._task_id is not None and self._base_url is not None:
            with contextlib.suppress(Exception):
                self._http("close", task_id=self._task_id)
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.stop(timeout=5)
            with contextlib.suppress(Exception):
                self._container.remove(force=True)
        self._container = None
        self._base_url = None

    async def close(self) -> None:
        self._teardown_container()

    async def shutdown(self) -> None:
        await self.close()


@dataclass
class AppWorldEnvConfig(BaseEnvConfig):
    """Configuration for :class:`AppWorldEnv`."""

    tool_class: ClassVar[type[RLEnvironment]] = AppWorldEnv
    image: str = "ghcr.io/stonybrooknlp/appworld:latest"
    data_root: str = ""
    in_container_root: str = "/run"
    server_port: int = 8000
    max_interactions: int = 50
    timeout_seconds: int = 120
    startup_timeout: int = 180
    dense_reward: bool = True
    mem_limit: str = "4g"
    experiment_prefix: str = "appworld_rl"
    backend: str = "docker"
    publish_port: bool = False
    docker_host: str | None = None
