"""AppWorld environment integration for GRPO rollouts."""

import contextlib
import importlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar

from openenv.core.env_server.types import State

from open_instruct import logger_utils

from .base import BaseEnvConfig, EnvCall, RLEnvironment, StepResult
from .tools.utils import coerce_args

logger = logger_utils.setup_logger(__name__)

_APPWORLD_EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "appworld_execute",
        "description": (
            "Execute Python code in an AppWorld task shell. The shell is stateful across calls. "
            "Use the injected `apis` object for app API calls and call "
            "`apis.supervisor.complete_task()` when finished."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute in the current AppWorld task context.",
                }
            },
            "required": ["code"],
        },
    },
}

_SCORE_KEYS = ("score", "task_goal_completion", "tgc", "sgc", "success", "completed", "task_completed")


def _load_appworld_symbols() -> tuple[type, Any]:
    """Lazily import appworld and return key symbols."""
    try:
        module = importlib.import_module("appworld")
    except ImportError as e:
        raise ImportError(
            "AppWorldEnv requires the `appworld` package. Install with `pip install appworld`, "
            "then run `appworld install` and `appworld download data`."
        ) from e

    appworld_cls = getattr(module, "AppWorld", None)
    if appworld_cls is None:
        raise ImportError("Could not find `AppWorld` in the installed appworld package.")
    update_root = getattr(module, "update_root", None)
    return appworld_cls, update_root


class AppWorldEnv(RLEnvironment):
    """Stateful AppWorld environment exposed as a single code-exec tool."""

    config_name = "appworld"
    _tool_definitions = (_APPWORLD_EXECUTE_TOOL,)

    def __init__(
        self,
        experiment_name: str = "open_instruct_appworld",
        appworld_root: str | None = None,
        raise_on_failure: bool = True,
        evaluate_on_done: bool = True,
        reward_scale: float = 1.0,
        penalty: float = -0.05,
        init_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        # EnvironmentPool always passes call_name; stateful envs do not use it.
        kwargs.pop("call_name", None)
        if kwargs:
            logger.warning(f"Ignoring unexpected AppWorldEnv kwargs: {sorted(kwargs.keys())}")

        self._appworld_cls, self._update_root = _load_appworld_symbols()

        self._default_experiment_name = experiment_name
        self._default_appworld_root = appworld_root
        self._default_raise_on_failure = raise_on_failure
        self._evaluate_on_done = evaluate_on_done
        self._reward_scale = reward_scale
        self._penalty = penalty
        self._default_init_kwargs = dict(init_kwargs or {})

        self._world: Any | None = None
        self._task_id: str | None = None
        self._step_count = 0
        self._completed = False
        self._last_eval_score: float | None = None

    @classmethod
    def get_tool_definitions(cls) -> list[dict]:
        return list(cls._tool_definitions)

    async def reset(self, **kwargs: Any) -> tuple[StepResult, list[dict]]:
        task_id = str(kwargs.get("task_id", "")).strip()
        if not task_id:
            raise ValueError("AppWorldEnv requires `task_id` in env_config.")

        await self.close()

        appworld_root = kwargs.get("appworld_root", self._default_appworld_root)
        if appworld_root:
            if self._update_root is not None:
                self._update_root(appworld_root)
            else:
                os.environ["APPWORLD_ROOT"] = str(appworld_root)

        experiment_name = str(kwargs.get("experiment_name", self._default_experiment_name))
        raise_on_failure = bool(kwargs.get("raise_on_failure", self._default_raise_on_failure))

        init_kwargs = dict(self._default_init_kwargs)
        override_init_kwargs = kwargs.get("appworld_init_kwargs")
        if isinstance(override_init_kwargs, dict):
            init_kwargs.update(override_init_kwargs)

        self._world = self._appworld_cls(
            task_id=task_id, experiment_name=experiment_name, raise_on_failure=raise_on_failure, **init_kwargs
        )
        self._task_id = task_id
        self._step_count = 0
        self._completed = False
        self._last_eval_score = None

        task = getattr(self._world, "task", None)
        instruction = str(getattr(task, "instruction", "")).strip()
        supervisor = getattr(task, "supervisor", {})
        supervisor_name = ""
        if isinstance(supervisor, dict):
            first_name = str(supervisor.get("first_name", "")).strip()
            last_name = str(supervisor.get("last_name", "")).strip()
            supervisor_name = " ".join(part for part in (first_name, last_name) if part)

        lines = [f"AppWorld task loaded: {task_id}"]
        if instruction:
            lines.append(f"Instruction: {instruction}")
        if supervisor_name:
            lines.append(f"Supervisor: {supervisor_name}")
        lines.append("Use `appworld_execute` to run Python code. Call `apis.supervisor.complete_task()` when done.")

        return StepResult(result="\n".join(lines)), list(self._tool_definitions)

    async def step(self, call: EnvCall) -> StepResult:
        if self._world is None:
            raise RuntimeError("Environment not reset. Call reset() first.")

        self._step_count += 1

        if call.name != "appworld_execute":
            return StepResult(
                result=f"Error: Unknown tool '{call.name}'. Available: appworld_execute",
                reward=self._penalty,
                metadata={"error": f"Unknown tool: {call.name}"},
            )

        args = coerce_args(_APPWORLD_EXECUTE_TOOL["function"]["parameters"], call.args)
        code = args.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return StepResult(
                result="Error: `code` must be a non-empty string.",
                reward=self._penalty,
                metadata={"error": "Empty code."},
            )

        start_time = time.perf_counter()
        try:
            output = self._world.execute(code)
        except Exception as e:
            runtime = time.perf_counter() - start_time
            return StepResult(
                result=f"AppWorld execution failed: {e}",
                reward=self._penalty,
                metadata={"error": str(e), "runtime": runtime},
            )

        runtime = time.perf_counter() - start_time
        output_text = "" if output is None else str(output)

        done = bool(self._world.task_completed())
        reward = 0.0
        metadata: dict[str, Any] = {"runtime": runtime, "error": ""}

        if done:
            self._completed = True
            evaluation_summary, evaluation_score, evaluation_error = self._evaluate_if_enabled()
            if evaluation_summary:
                output_text = f"{output_text}\n\n{evaluation_summary}" if output_text else evaluation_summary
            if evaluation_error:
                metadata["error"] = evaluation_error

            if evaluation_score is not None:
                self._last_eval_score = evaluation_score
                metadata["evaluation_score"] = evaluation_score
                reward = evaluation_score * self._reward_scale
            else:
                reward = 0.0 if self._evaluate_on_done else self._reward_scale

        return StepResult(result=output_text, reward=reward, done=done, metadata=metadata)

    def _evaluate_if_enabled(self) -> tuple[str, float | None, str]:
        if not self._evaluate_on_done or self._world is None:
            return "", None, ""

        try:
            evaluation = self._world.evaluate()
            evaluation_dict = evaluation.to_dict() if hasattr(evaluation, "to_dict") else {}
            score = self._extract_score(evaluation_dict)
            if score is None:
                return "AppWorld evaluation complete.", None, ""
            return f"AppWorld evaluation score: {score:.4f}", score, ""
        except Exception as e:
            return "", None, f"AppWorld evaluation failed: {e}"

    def _extract_score(self, value: Any) -> float | None:
        scalar = self._coerce_scalar(value)
        if scalar is not None:
            return scalar

        if isinstance(value, dict):
            for key in _SCORE_KEYS:
                if key in value:
                    found = self._coerce_scalar(value[key])
                    if found is not None:
                        return found

            passes = value.get("passes")
            fails = value.get("fails")
            if isinstance(passes, list) and isinstance(fails, list):
                total = len(passes) + len(fails)
                if total > 0:
                    return len(passes) / total

            num_passed = self._coerce_scalar(value.get("num_passed_tests"))
            num_failed = self._coerce_scalar(value.get("num_failed_tests"))
            if num_passed is not None and num_failed is not None and (num_passed + num_failed) > 0:
                return num_passed / (num_passed + num_failed)

            for nested in value.values():
                found = self._extract_score(nested)
                if found is not None:
                    return found

        if isinstance(value, list):
            for nested in value:
                found = self._extract_score(nested)
                if found is not None:
                    return found

        return None

    @staticmethod
    def _coerce_scalar(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        return None

    def get_metrics(self) -> dict[str, float]:
        metrics = {"step_count": float(self._step_count), "completed": 1.0 if self._completed else 0.0}
        if self._last_eval_score is not None:
            metrics["evaluation_score"] = float(self._last_eval_score)
        return metrics

    def state(self) -> State:
        return State(episode_id=self._task_id, step_count=self._step_count)

    async def close(self) -> None:
        if self._world is not None:
            with contextlib.suppress(Exception):
                self._world.close()
            self._world = None

    async def shutdown(self) -> None:
        """Clean up world resources on actor shutdown."""
        await self.close()


@dataclass
class AppWorldEnvConfig(BaseEnvConfig):
    """Configuration for AppWorldEnv."""

    tool_class: ClassVar[type[RLEnvironment]] = AppWorldEnv
    experiment_name: str = "open_instruct_appworld"
    appworld_root: str | None = None
    raise_on_failure: bool = True
    evaluate_on_done: bool = True
    reward_scale: float = 1.0
    penalty: float = -0.05
    init_kwargs: dict[str, Any] = field(default_factory=dict)
