"""H068 teacher-free configuration, allocation and parent-compiler contracts."""

import copy
import dataclasses
import difflib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import tomllib

from open_instruct.miles.configuration import topology
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.distillation import config
from open_instruct.miles.execution import cluster

REPO = Path(__file__).resolve().parents[1]
PARENT = "024e2dfa2eee1a3d8aaef00689422e11acba261e"
CONFIGS = REPO / "runs/h068"
IDENTITY = {
    "wandb_run_name": "<run>",
    "save": "<root>/checkpoints",
    "save_debug_rollout_data": "<root>/rollouts/{rollout_id}.pt",
    "wandb_dir": "<root>/wandb",
}


def load(arm):
    return RunSpec.load(CONFIGS / f"{arm}-s1.toml")


def prepared(spec):
    return {
        "hf_checkpoint": spec.model["source"],
        "prompt_data": spec.data["prompt_data"],
        "reward_config": spec.data["reward_config"],
        "eval_prompt_data": [],
    }


def maps(spec):
    compiled = spec.compile(prepared(spec))
    return {"core": dataclasses.asdict(compiled.core), "miles": compiled.resolved_miles()}


def canonicalize(raw, spec):
    result = copy.deepcopy(raw)
    for key, replacement in IDENTITY.items():
        expected = spec.name if key == "wandb_run_name" else replacement.replace("<root>", spec.output["root"])
        assert result["miles"][key] == expected
        result["miles"][key] = replacement
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


@pytest.fixture(scope="module")
def parent_compiler(tmp_path_factory):
    root = tmp_path_factory.mktemp("h068-parent")
    archive = subprocess.check_output(
        ["git", "archive", PARENT, "open_instruct/miles", "open_instruct/__init__.py"], cwd=REPO
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(root, filter="data")
    return root


@pytest.mark.parametrize("arm", ["opd", "rl-cont"])
def test_h068_existing_arms_compile_identically_to_registered_parent(parent_compiler, arm):
    spec = load(arm)
    script = """
import dataclasses, json, sys
from open_instruct.miles.configuration import topology
from open_instruct.miles.configuration.run_spec import RunSpec
spec = RunSpec.load(sys.argv[1])
compiled = spec.compile(json.loads(sys.argv[2]))
print(json.dumps({'core': dataclasses.asdict(compiled.core), 'miles': compiled.resolved_miles(),
                  'teacher': spec.teacher, 'allocation': topology.plan(spec)}))
"""
    baseline = json.loads(
        subprocess.check_output(
            [sys.executable, "-c", script, str(spec.config_path), json.dumps(prepared(spec))],
            cwd=parent_compiler,
            env=dict(os.environ, PYTHONPATH=str(parent_compiler)),
            text=True,
        )
    )
    current = maps(spec) | {"teacher": spec.teacher, "allocation": topology.plan(spec)}
    # Retain raw and canonical maps for failures; exclude exactly the four registered values.
    for label, raw in (("parent", baseline), ("child", current)):
        (parent_compiler / f"{arm}-{label}-raw.json").write_text(json.dumps(raw, sort_keys=True, indent=2))
        (parent_compiler / f"{arm}-{label}-canonical.json").write_text(canonicalize(raw, spec))
    (parent_compiler / f"{arm}.diff").write_text(
        "\n".join(
            difflib.unified_diff(canonicalize(baseline, spec).splitlines(), canonicalize(current, spec).splitlines())
        )
    )
    assert canonicalize(current, spec) == canonicalize(baseline, spec)


def test_tailneg_differs_from_opd_only_in_objective_service_and_student_gpus():
    opd, tail = load("opd"), load("tailneg")
    assert prepared(opd) == prepared(tail) == prepared(load("rl-cont"))
    raw = maps(tail)
    assert raw["miles"]["rollout_num_gpus"] == 6
    raw["miles"]["rollout_num_gpus"] = 4
    assert canonicalize(raw, tail) == canonicalize(maps(opd), opd)
    assert tail.sections["training"]["filter_zero_std_groups"] is False
    assert not tail.compile().core.filter_zero_std_groups
    assert tail.compile().miles["rollout_batch_size"] == 64
    assert tail.compile().miles["n_samples_per_prompt"] == 8
    assert tail.compile().miles["global_batch_size"] == 512
    assert tail.teacher["kl_coef"] == 1
    assert tail.teacher["tail_fraction"] == 0.2
    assert tail.teacher["tail_value"] == 0.5
    assert not config.requires_teacher(tail.teacher)
    assert config.requires_teacher(opd.teacher)
    assert RunSpec.from_dict(tail.to_dict()).teacher == tail.teacher


@pytest.mark.parametrize(
    "arm,replicas,trainer,student,teacher", [("opd", 1, 2, 4, 2), ("tailneg", 1, 2, 6, 0), ("rl-cont", 2, 4, 12, 0)]
)
def test_registered_allocations(arm, replicas, trainer, student, teacher):
    layout = topology.plan(load(arm))
    assert layout["replicas"] == replicas
    assert layout["allocated_gpus"] == replicas * 8
    assert sum(node["trainer_gpus"] for node in layout["nodes"]) == trainer
    assert sum(node["rollout_gpus"] for node in layout["nodes"]) == student
    assert layout["teacher_gpus"] == teacher
    assert layout["unused_gpus"] == 0
    if arm == "rl-cont":
        assert [(node["trainer_gpus"], node["rollout_gpus"]) for node in layout["nodes"]] == [(4, 4), (0, 8)]


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "/teacher"),
        ("gpus", 1),
        ("gpus", False),
        ("tensor_parallel_size", 1),
        ("concurrency", 4),
        ("pressure_clip", 1),
        ("calibrate_initial", True),
        ("tail_fraction", 0),
        ("tail_fraction", 1.01),
        ("tail_fraction", float("nan")),
        ("tail_fraction", True),
        ("tail_value", 0),
        ("tail_value", float("inf")),
        ("kl_coef", 0),
    ],
)
def test_tailneg_rejects_teacher_resources_and_invalid_parameters(field, value):
    teacher = {"objective": "tail_negative", field: value}
    with pytest.raises(ValueError):
        config.parse({"teacher": teacher}, Path("/tmp"))


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("training", "filter_zero_std_groups", True),
        ("launch", "auto_resume", True),
        ("miles", "normalize_advantages", True),
        ("miles", "use_rollout_logprobs", True),
        ("miles", "opd_kl_coef", 2),
        ("miles", "kl_coef", 1),
        ("miles", "rollout_top_p", 0.9),
        ("async", "fully_async", True),
        ("core", "publication_mode", "refresh"),
    ],
)
def test_tailneg_keeps_opd_refusal_paths(section, field, value):
    document = tomllib.loads((CONFIGS / "tailneg-s1.toml").read_text())
    document.setdefault(section, {})[field] = value
    with pytest.raises(ValueError):
        RunSpec.from_dict(document)


@pytest.mark.parametrize("objective", ["full", "cancel_answer_positive", "typo"])
def test_tail_fields_do_not_leak_into_teacher_objectives(objective):
    with pytest.raises(ValueError):
        config.parse({"teacher": {"source": "/teacher", "objective": objective, "tail_fraction": 0.2}}, Path("/tmp"))


def test_tailneg_cluster_starts_no_teacher_and_propagates_objective(tmp_path, monkeypatch):
    spec = load("tailneg")
    spec.output["root"] = str(tmp_path)
    started = {}

    class StartedRay(Exception):
        pass

    class Supervisor:
        def __init__(self, *args):
            self.thread = SimpleNamespace(start=lambda: None)
            self.health = {}

        def wait(self, predicate, **kwargs):
            assert predicate()

        def start(self, label, command, env):
            started[label] = env
            assert label == "ray"
            raise StartedRay

        def run_child(self, *args):
            pytest.fail("No teacher or teacher probe may start")

        def close(self):
            pass

    monkeypatch.setattr(cluster.RunSpec, "load", lambda path: spec)
    monkeypatch.setattr(cluster.rendezvous, "join", lambda *args: tmp_path)
    monkeypatch.setattr(cluster.socket, "gethostbyname", lambda *args: "10.0.0.1")
    monkeypatch.setattr(cluster.signal, "signal", lambda *args: None)
    monkeypatch.setattr(cluster, "Supervisor", Supervisor)
    monkeypatch.setattr(cluster, "free_ports", lambda count: list(range(20000, 20000 + count)))
    monkeypatch.setenv("OI_MILES_REPLICA_RANK", "0")
    monkeypatch.setenv("OI_MILES_REPLICA_COUNT", "1")
    monkeypatch.setenv("OI_MILES_LAUNCH_ID", "test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    with pytest.raises(StartedRay):
        cluster.run("unused.toml")
    assert list(started) == ["ray"]
    assert started["ray"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert json.loads(started["ray"][config.ENV]) == spec.teacher
    assert not (tmp_path / "teacher.json").exists()


def test_tail_count_is_exact_ceiling():
    for length in range(1, 5001):
        pressure = hooks.tail_negative([-1.0] * length, length, 0.2, 0.5)
        assert sum(1 for value in pressure if value) == -(-length // 5)
    assert sum(hooks.tail_negative([-1.0] * 10, 10, 0.3, 0.5)) == 0.5 * 3
