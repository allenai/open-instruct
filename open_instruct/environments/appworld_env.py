"""AppWorld environment for interactive coding agent tasks."""

import logging
from typing import Any

from .base import ResetResult, RLEnvironment, StepResult, ToolCall, register_env

try:
    from appworld import AppWorld

    HAS_APPWORLD = True
except ImportError:
    AppWorld = None
    HAS_APPWORLD = False

logger = logging.getLogger(__name__)


@register_env("appworld")
class AppWorldEnv(RLEnvironment):
    """
    AppWorld environment using raw code execution.

    The model generates Python code that calls APIs via `apis.{app}.{api}(**params)`.
    See: https://github.com/stonybrooknlp/appworld
    """

    use_tool_calls: bool = True
    response_role: str = "tool"
    max_steps: int = 50

    def __init__(
        self,
        experiment_name: str = "rl_training",
        max_interactions: int = 100,
        max_api_calls_per_interaction: int = 500,
        raise_on_failure: bool = False,
        random_seed: int = 42,
        ground_truth_mode: str = "none",
        **kwargs: Any,
    ):
        self._experiment_name = experiment_name
        self._max_interactions = max_interactions
        self._max_api_calls_per_interaction = max_api_calls_per_interaction
        self._raise_on_failure = raise_on_failure
        self._random_seed = random_seed
        self._ground_truth_mode = ground_truth_mode
        self._extra_kwargs = kwargs

        self._world = None
        self._task_id: str | None = None
        self._step_count = 0
        self._completed = False
        self._evaluation_result: dict | None = None

    async def reset(self, task_id: str | None = None) -> ResetResult:
        """Initialize AppWorld for a task."""
        if not HAS_APPWORLD:
            raise ImportError(
                "appworld not installed. Run:\n  pip install appworld\n  appworld install\n  appworld download data"
            )

        if self._world is not None:
            self._world.close()

        if task_id is None:
            raise ValueError("task_id is required for AppWorld")

        self._task_id = task_id
        self._step_count = 0
        self._completed = False
        self._evaluation_result = None

        self._world = AppWorld(
            task_id=task_id,
            experiment_name=self._experiment_name,
            max_interactions=self._max_interactions,
            max_api_calls_per_interaction=self._max_api_calls_per_interaction,
            raise_on_failure=self._raise_on_failure,
            random_seed=self._random_seed,
            ground_truth_mode=self._ground_truth_mode,
            **self._extra_kwargs,
        )

        task = self._world.task
        observation = f"""Task: {task.instruction}

Supervisor: {task.supervisor["first_name"]} {task.supervisor["last_name"]}
Email: {task.supervisor["email"]}

Available Apps: {", ".join(task.app_descriptions.keys())}

You can execute Python code using the `execute` tool. Use `apis.{{app_name}}.{{api_name}}(**params)` to call APIs.
Use `apis.api_docs.show_api_doc(app_name, api_name)` to look up API documentation.
Use `apis.supervisor.complete_task()` when done (with `answer=` if the task requires an answer)."""

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "execute",
                    "description": "Execute Python code in the AppWorld environment. Use apis.{app}.{api}(**params) to call APIs.",
                    "parameters": {
                        "type": "object",
                        "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                        "required": ["code"],
                    },
                },
            }
        ]

        return ResetResult(
            observation=observation,
            tools=tools,
            info={"task_id": task_id, "supervisor": task.supervisor, "apps": list(task.app_descriptions.keys())},
        )

    async def step(self, tool_call: ToolCall) -> StepResult:
        """Execute code in AppWorld."""
        if self._world is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if tool_call.name != "execute":
            return StepResult(
                observation=f"Unknown tool: {tool_call.name}. Use 'execute' with Python code.", reward=0.0, done=False
            )

        code = tool_call.args.get("code", "")
        if not code.strip():
            return StepResult(observation="Error: Empty code provided.", reward=0.0, done=False)

        try:
            output = self._world.execute(code)
        except Exception as e:
            output = f"Execution error: {e}"

        done = self._world.task_completed()
        reward = 0.0

        if done:
            self._completed = True
            try:
                evaluation = self._world.evaluate()
                self._evaluation_result = evaluation.to_dict()
                passes = len(self._evaluation_result.get("passes", []))
                fails = len(self._evaluation_result.get("fails", []))
                total = passes + fails
                reward = passes / total if total > 0 else 0.0
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}")
                reward = 0.0

        return StepResult(
            observation=output if output else "(no output)",
            reward=reward,
            done=done,
            info={"step_count": self._step_count, "completed": done},
        )

    def get_metrics(self) -> dict[str, float]:
        """Return AppWorld-specific metrics."""
        metrics = {"step_count": float(self._step_count), "completed": 1.0 if self._completed else 0.0}

        if self._evaluation_result:
            passes = len(self._evaluation_result.get("passes", []))
            fails = len(self._evaluation_result.get("fails", []))
            metrics["tests_passed"] = float(passes)
            metrics["tests_failed"] = float(fails)
            metrics["test_pass_rate"] = passes / (passes + fails) if (passes + fails) > 0 else 0.0

        return metrics

    async def close(self) -> None:
        """Close the AppWorld instance."""
        if self._world is not None:
            self._world.close()
            self._world = None
