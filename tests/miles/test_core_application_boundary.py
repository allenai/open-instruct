"""CPU planning/artifact readers and the runtime must keep shared semantics."""

import ast
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from miles.backends import core_utils
from miles.backends.core_utils import actor, performance, scoring, timing
from miles.backends.core_utils.rollout import async_buffer
from miles.utils.function_registry import load_function
from scripts.miles import capacity_metrics

from open_instruct.miles.configuration import config
from open_instruct.miles.evaluation import evaluation


@pytest.mark.parametrize(
    "runtime,application,names",
    [
        ("async_capacity.py", "configuration/async_capacity.py", None),
        ("infra_timeouts.py", "infrastructure/infra_timeouts.py", None),
        ("publication/state.py", "infrastructure/artifacts.py", {"atomic_json"}),
        ("scoring.py", "configuration/config.py", {"ScoringPass", "scoring_pass"}),
        ("validation.py", "configuration/validation.py", {"integer", "number", "mapping", "fields"}),
    ],
)
def test_cpu_boundary_algorithms_match_runtime(runtime, application, names):
    root = Path(__file__).resolve().parents[2] / "open_instruct/miles"

    def definitions(path):
        result = {}
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if names is not None and node.name not in names:
                continue
            for child in ast.walk(node):
                if isinstance(child, ast.arg):
                    child.annotation = None
            result[node.name] = ast.dump(node)
        return result

    assert definitions(Path(core_utils.__file__).parent / runtime) == definitions(root / application)


def test_offline_rates_match_runtime():
    assert ast.dump(ast.parse(inspect.getsource(performance.training_rates))) == ast.dump(
        ast.parse(inspect.getsource(capacity_metrics.training_rates))
    )


def test_recipe_and_runtime_scoring_agree():
    core = config.CoreConfig()
    options = dict(global_batch_size=4, rollout_batch_size=1, n_samples_per_prompt=4)
    assert scoring.scoring_pass(core, options).as_dict() == config.scoring_pass(core, options).as_dict()


def test_snapshot_callback_preserves_schedule_and_output(tmp_path):
    settings = dict(
        root=str(tmp_path), initial=False, final=True, total_updates=3, tasks=[dict(interval=2, generation={})]
    )
    args = SimpleNamespace(background_evaluation=settings)
    callback = load_function("open_instruct.miles.evaluation.evaluation.training_snapshot")
    assert callback(args, 1) is None
    for update in (2, 3):
        assert callback(args, update) == str(evaluation.snapshot(tmp_path, update))


def test_backend_modules_have_no_application_imports():
    for path in Path(core_utils.__file__).parent.rglob("*.py"):
        assert "open_instruct" not in path.read_text(), path
    assert actor.OLMoCoreTrainRayActor.__module__ == "miles.backends.core_utils.actor"
    assert async_buffer.HomogeneousPolicyDataBuffer.__module__ == "miles.backends.core_utils.rollout.async_buffer"


def test_startup_timer_records_success_and_failure(tmp_path):
    args = SimpleNamespace(save=str(tmp_path), rank=2)
    device = mock.Mock()
    with timing.startup_stage(args, "build", device=device):
        pass
    with pytest.raises(ValueError), timing.startup_stage(args, "restore"):
        raise ValueError("injected")
    device.synchronize.assert_called_once()
    rows = [json.loads(s) for s in (tmp_path / "startup_rank2.jsonl").read_text().splitlines()]
    assert [r["passed"] for r in rows] == [True, False]
    assert all(r["seconds"] >= 0 for r in rows)
