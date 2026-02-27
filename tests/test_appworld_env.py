"""Unit tests for AppWorld environment integration."""

import asyncio
import types
import unittest
from unittest.mock import MagicMock, patch

import open_instruct.environments.appworld as appworld_module
from open_instruct.environments.appworld import AppWorldEnv, AppWorldEnvConfig
from open_instruct.environments.base import EnvCall, StepResult
from open_instruct.environments.tools.tools import TOOL_REGISTRY


class _FakeEvaluation:
    def __init__(self, score: float, success: bool):
        self._score = score
        self._success = success

    def to_dict(self) -> dict:
        return {"score": self._score, "success": self._success}


class _FakeTask:
    instruction = "Move money to savings."
    supervisor = {"first_name": "Alex", "last_name": "Rivera"}


class _FakeAppWorld:
    instances: list["_FakeAppWorld"] = []
    evaluation_score: float = 0.75

    def __init__(self, task_id: str, experiment_name: str, raise_on_failure: bool = True, **kwargs):
        self.task_id = task_id
        self.experiment_name = experiment_name
        self.raise_on_failure = raise_on_failure
        self.kwargs = kwargs
        self.task = _FakeTask()
        self.executed: list[str] = []
        self._completed = False
        self.closed = False
        _FakeAppWorld.instances.append(self)

    def execute(self, code: str) -> str:
        self.executed.append(code)
        if "complete_task" in code:
            self._completed = True
            return "Task marked complete."
        return "ok"

    def task_completed(self) -> bool:
        return self._completed

    def evaluate(self) -> _FakeEvaluation:
        return _FakeEvaluation(score=self.evaluation_score, success=self._completed)

    def close(self) -> None:
        self.closed = True


class TestAppWorldEnv(unittest.TestCase):
    def setUp(self):
        _FakeAppWorld.instances = []
        _FakeAppWorld.evaluation_score = 0.75

    def _mock_module(self, update_root: MagicMock | None = None):
        return types.SimpleNamespace(
            AppWorld=_FakeAppWorld, update_root=update_root if update_root is not None else lambda _root: None
        )

    def _patch_appworld_available(self, update_root: MagicMock | None = None):
        return patch.multiple(
            appworld_module,
            APPWORLD_AVAILABLE=True,
            _APPWORLD_MODULE=self._mock_module(update_root=update_root),
            APPWORLD_IMPORT_ERROR=None,
        )

    def test_appworld_registered(self):
        self.assertIn("appworld", TOOL_REGISTRY)
        self.assertEqual(TOOL_REGISTRY["appworld"], AppWorldEnvConfig)

    def test_reset_requires_task_id(self):
        with self._patch_appworld_available():
            env = AppWorldEnv()
            with self.assertRaises(ValueError):
                asyncio.run(env.reset())

    def test_reset_returns_tools_and_observation(self):
        with self._patch_appworld_available():
            env = AppWorldEnv(experiment_name="unit-test-exp")
            result, tools = asyncio.run(env.reset(task_id="task-001"))

        self.assertIsInstance(result, StepResult)
        self.assertIn("AppWorld task loaded", result.result)
        self.assertEqual([t["function"]["name"] for t in tools], ["appworld_execute"])
        self.assertEqual(_FakeAppWorld.instances[-1].task_id, "task-001")
        self.assertEqual(_FakeAppWorld.instances[-1].experiment_name, "unit-test-exp")

    def test_step_and_completion_reward(self):
        with self._patch_appworld_available():
            env = AppWorldEnv(reward_scale=1.0)
            asyncio.run(env.reset(task_id="task-002"))

            first = asyncio.run(env.step(EnvCall(id="1", name="appworld_execute", args={"code": "print('hello')"})))
            done = asyncio.run(
                env.step(EnvCall(id="2", name="appworld_execute", args={"code": "apis.supervisor.complete_task()"}))
            )

        self.assertFalse(first.done)
        self.assertEqual(first.reward, 0.0)
        self.assertTrue(done.done)
        self.assertAlmostEqual(done.reward, _FakeAppWorld.evaluation_score)
        self.assertIn("evaluation score", done.result.lower())
        metrics = env.get_metrics()
        self.assertEqual(metrics["completed"], 1.0)
        self.assertAlmostEqual(metrics["evaluation_score"], _FakeAppWorld.evaluation_score)

    def test_unknown_tool_returns_penalty(self):
        with self._patch_appworld_available():
            env = AppWorldEnv(penalty=-0.2)
            asyncio.run(env.reset(task_id="task-003"))
            result = asyncio.run(env.step(EnvCall(id="1", name="not_a_tool", args={})))

        self.assertFalse(result.done)
        self.assertEqual(result.reward, -0.2)
        self.assertIn("Unknown tool", result.result)

    def test_appworld_root_calls_update_root(self):
        update_root = MagicMock()
        with self._patch_appworld_available(update_root=update_root):
            env = AppWorldEnv(appworld_root="/tmp/appworld")
            asyncio.run(env.reset(task_id="task-004"))
        update_root.assert_called_once_with("/tmp/appworld")

    def test_close_closes_world(self):
        with self._patch_appworld_available():
            env = AppWorldEnv()
            asyncio.run(env.reset(task_id="task-005"))
            world = _FakeAppWorld.instances[-1]
            asyncio.run(env.close())

        self.assertTrue(world.closed)

    def test_guard_raises_clean_error_when_dependency_missing(self):
        with patch.multiple(
            appworld_module,
            APPWORLD_AVAILABLE=False,
            _APPWORLD_MODULE=None,
            APPWORLD_IMPORT_ERROR=ImportError("No module named appworld"),
        ), self.assertRaises(ImportError) as ctx:
            appworld_module._load_appworld_symbols()
        self.assertIn("optional `appworld` dependency", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
