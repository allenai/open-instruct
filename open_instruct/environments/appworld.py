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
            "Task context should come from the dataset prompt because env reset observations "
            "are not injected back into the model conversation. "
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

_APPWORLD_MODULE: Any | None = None
APPWORLD_AVAILABLE = False
APPWORLD_IMPORT_ERROR: ImportError | None = None
try:
    _APPWORLD_MODULE = importlib.import_module("appworld")
    APPWORLD_AVAILABLE = True
except ImportError as e:
    APPWORLD_IMPORT_ERROR = e


def _load_appworld_symbols() -> type:
    """Return appworld symbols if dependency is available."""
    if not APPWORLD_AVAILABLE or _APPWORLD_MODULE is None:
        message = (
            "AppWorldEnv requires the optional `appworld` dependency. Install with "
            "`pip install -e '.[appworld]'` (or `pip install appworld`), then run "
            "`appworld install` and `appworld download data`."
        )
        if APPWORLD_IMPORT_ERROR is not None:
            raise ImportError(message) from APPWORLD_IMPORT_ERROR
        raise ImportError(message)

    module = _APPWORLD_MODULE
    appworld_cls = getattr(module, "AppWorld", None)
    if appworld_cls is None:
        raise ImportError("Could not find `AppWorld` in the installed appworld package.")
    return appworld_cls


def is_appworld_available() -> bool:
    """Return True when the optional appworld dependency is importable."""
    return APPWORLD_AVAILABLE


class AppWorldEnv(RLEnvironment):
    """Stateful AppWorld environment exposed as a single code-exec tool.

    Note: the rollout pipeline does not inject reset() observations into the
    model conversation, so task instructions/context must be present in the
    dataset prompt.
    """

    config_name = "appworld"
    _tool_definitions = (_APPWORLD_EXECUTE_TOOL,)

    def __init__(
        self,
        experiment_name: str = "open_instruct_appworld",
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

        self._appworld_cls = _load_appworld_symbols()

        self._default_experiment_name = experiment_name
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

        if kwargs.get("appworld_root") is not None:
            raise ValueError(
                "`appworld_root` in env_config is not supported. Set `APPWORLD_ROOT` "
                "before launching training/evaluation."
            )
        if not os.environ.get("APPWORLD_ROOT"):
            raise ValueError(
                "`APPWORLD_ROOT` is required for AppWorldEnv and must be set by the caller before creating rollouts."
            )

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
        return StepResult(result="", metadata={"task_id": task_id}), list(self._tool_definitions)

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
    raise_on_failure: bool = True
    evaluate_on_done: bool = True
    reward_scale: float = 1.0
    penalty: float = -0.05
    init_kwargs: dict[str, Any] = field(default_factory=dict)
