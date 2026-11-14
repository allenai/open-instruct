"""Native BFCL dispatch, checkpoint ownership, full-category validity and resume boundaries."""

import copy
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import bfcl_runtime as bfcl
from open_instruct.miles.evaluation import evaluation
from open_instruct.miles.evaluation import evaluation_runner as runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def run():
    return RunSpec.load(ROOT / "configs/miles/examples/tiny/three-node.toml")


@pytest.fixture
def receipt(run, tmp_path):
    config = evaluation.runtime(run, run.compile().miles)
    config["root"] = str(tmp_path / "run")
    tasks = next(group for group in evaluation.groups(config, 0, 200) if group[0].get("worker") == bfcl.WORKER)
    return {
        "checkpoint": str(tmp_path / "checkpoint"),
        "update": 100,
        "group_id": "native-bfcl",
        "tasks": tasks,
        "evaluation": bfcl.settings(config),
        "training": {"name": "test-bfcl", "wandb": {"mode": "offline"}},
        "runner_sha256": hashlib.sha256(Path(runner.__file__ or "").read_bytes()).hexdigest(),
    }


def test_worker_groups_and_images_are_independent(run, tmp_path):
    config = evaluation.runtime(run, run.compile().miles)
    groups = evaluation.groups(config, 0, 200)
    assert len(run.evaluation["tasks"]) == 42 and len(groups) == 8
    native = [group for group in groups if group[0].get("worker")]
    assert sorted(map(len, native)) == [4, 13]
    assert len([task for task in run.evaluation["tasks"] if not task.get("worker")]) == 25
    for group in native:
        configured = bfcl.settings(config)
        assert configured["image"] == bfcl.PROFILE["image"]
        assert configured["revision"] == bfcl.PROFILE["evaluator_revision"]
        assert configured["worker"]["identity"] == group[0]["worker_profile"]
    assert config["image"] != bfcl.PROFILE["image"]  # Ordinary SG panel is not silently repinned.


@pytest.mark.parametrize("panel", ["single", "multi"])
def test_native_argv_uses_exact_checkpoint_tools_and_per_turn_budget(receipt, run, panel):
    receipt["tasks"] = [t for t in run.evaluation["tasks"] if t.get("worker") and bfcl.panel_for(t["task"]) == panel]
    script = bfcl.launch_command(receipt, Path("/weka/test/native"))
    argv = shlex.split(script.splitlines()[-1])
    assert argv[argv.index("--model") + 1] == receipt["checkpoint"]
    assert argv[argv.index("--harness") + 1] == bfcl.PROFILE["panels"][panel]["harness"]
    assert argv.count("--task") == (13 if panel == "single" else 4)
    options = [argv[i + 1] for i, value in enumerate(argv) if value == "-o"]
    for option in [
        "provider.kind=vllm_server",
        "provider.kwargs.tool_call_parser=qwen3_xml",
        "provider.kwargs.reasoning_parser=olmo3",
        "provider.kwargs.max_model_len=65536",
        "provider.kwargs.mamba_ssm_cache_dtype=float32",
        "provider.kwargs.enable_prefix_caching=false",
    ]:
        assert option in options
    for option in [
        "top_p=0.95",
        "temperature=1.0",
        "fit_max_tokens_to_context=true",
        "thinking_prefilled=false",
        "strip_thinking=true",
        "strip_unclosed_thinking=true",
        f"max_tokens={4096 if panel == 'single' else 16384}",
    ]:
        assert options.count(option) == argv.count("--task")
    assert "--resume" in argv and "--fresh" not in argv
    assert "/robertb/" not in script and "robertb_RL_QUALIFICATION_HF_TOKEN" not in script
    assert "/results" not in script  # Existing MILES wrapper owns the non-mounted Beaker output.


def test_native_spec_freezes_worker_source_and_image(receipt):
    spec = evaluation.specification(receipt)
    job = spec["tasks"][0]
    assert job["image"] == {"beaker": bfcl.PROFILE["image"]}
    assert job["result"]["path"] == "/output"
    assert job["resources"]["gpuCount"] == 1
    assert len(job["arguments"][0].encode()) < 120 * 1024
    changed = copy.deepcopy(receipt)
    changed["evaluation"]["worker"]["identity"]["source_sha256"] = "0" * 64
    with pytest.raises(InputError, match="worker differs"):
        evaluation.specification(changed)


def test_native_tasks_cannot_fall_through_to_sglang(run):
    document = run.to_dict()
    task = next(t for t in document["evaluation"]["tasks"] if t.get("worker"))
    task.pop("worker")
    task.pop("worker_profile")
    with pytest.raises(InputError, match="dedicated BFCL worker"):
        RunSpec.from_dict(document)


def test_partial_category_panel_and_foreign_profile_rejected(run):
    document = run.to_dict()
    document["evaluation"]["tasks"] = [next(t for t in document["evaluation"]["tasks"] if t.get("worker"))]
    with pytest.raises(InputError, match="panel in full"):
        RunSpec.from_dict(document)
    document = run.to_dict()
    next(t for t in document["evaluation"]["tasks"] if t.get("worker"))["worker_profile"]["image"] = "foreign"
    with pytest.raises(InputError, match="profile differs"):
        RunSpec.from_dict(document)


def test_archive_locator_is_configurable_but_profile_remains_pinned(run):
    document = run.to_dict()
    document["evaluation"]["bfcl"]["plugin_archive"] = "/weka/shared/relocated-ladder.tar"
    for task in document["evaluation"]["tasks"]:
        task.pop("worker_profile", None)
    relocated = RunSpec.from_dict(document)
    settings = bfcl.settings(relocated.evaluation)
    assert settings["worker"]["profile"]["plugin_archive"] == "/weka/shared/relocated-ladder.tar"
    assert settings["worker"]["profile"]["plugin_archive_sha256"] == bfcl.PROFILE["plugin_archive_sha256"]
    assert (
        next(t for t in relocated.evaluation["tasks"] if t.get("worker"))["worker_profile"]
        != next(t for t in run.evaluation["tasks"] if t.get("worker"))["worker_profile"]
    )


def test_checkpoint_marker_foreign_model_and_template_fail_before_runtime(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    with pytest.raises(ValueError, match="completion marker"):
        bfcl.validate_checkpoint(checkpoint, 1, bfcl.PROFILE)
    (checkpoint / ".complete").touch()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "foreign"}))
    with pytest.raises(ValueError, match="model family"):
        bfcl.validate_checkpoint(checkpoint, 1, bfcl.PROFILE)
    (checkpoint / "config.json").write_text(json.dumps(bfcl.PROFILE["model_config"]))
    for name in bfcl.PROFILE["tokenizer_assets"]:
        (checkpoint / name).write_text("foreign")
    with pytest.raises(ValueError, match="tokenizer/tool template"):
        bfcl.validate_checkpoint(checkpoint, 1, bfcl.PROFILE)


def write_native_results(output, tasks, delta=0):
    rows = []
    for task in tasks:
        name = task["task"]
        panel = bfcl.panel_for(name)
        key = name.removeprefix("bfcl_multi_turn_") if panel == "multi" else name.removeprefix("bfcl_")
        rows.append(
            {
                "task": name,
                "instances_saved": bfcl.PROFILE["panels"][panel]["category_counts"][key] + delta,
                "metrics": {"bfcl_accuracy": {"mean": 0.5}},
            }
        )
    runner.write_json(output / "metrics.json", {"tasks": rows})
    (output / "native-predictions.jsonl").write_text(json.dumps({"model_output": [{"text": ""}]}) + "\n")


def test_full_native_category_counts_are_required(receipt, tmp_path):
    write_native_results(tmp_path, receipt["tasks"])
    bfcl.validate_counts(tmp_path, receipt["tasks"])
    write_native_results(tmp_path, receipt["tasks"], delta=-1)
    with pytest.raises(ValueError, match="populations differ"):
        bfcl.validate_counts(tmp_path, receipt["tasks"])


def test_native_failure_has_no_completion_claim_and_retry_resumes(receipt, tmp_path, monkeypatch):
    calls = []
    fail = [True]
    monkeypatch.setattr(runner, "native_bfcl_worker", lambda r: bfcl)
    monkeypatch.setattr(bfcl, "validate_checkpoint", lambda *args: None)
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *args, **kwargs: bfcl.PROFILE["evaluator_revision"])
    output = runner.result_directory(receipt)

    def native(argv, **kwargs):
        calls.append(argv)
        write_native_results(output / "native", receipt["tasks"])
        if fail[0]:
            raise subprocess.CalledProcessError(143, argv)

    monkeypatch.setattr(runner.subprocess, "run", native)
    copytree = runner.shutil.copytree

    def copy_results(src, dst, *args, **kwargs):
        if dst == "/output/evaluation":
            dst = tmp_path / "beaker-output"
        return copytree(src, dst, *args, **kwargs)

    monkeypatch.setattr(runner.shutil, "copytree", copy_results)
    with pytest.raises(subprocess.CalledProcessError):
        runner.run_evaluation(receipt)
    assert not (output / "native/worker-complete.json").exists()
    assert (tmp_path / "beaker-output/native/metrics.json").exists()
    fail[0] = False
    runner.run_evaluation(receipt)
    assert len(calls) == 2 and "--resume" in calls[1][2]
    assert (output / "native/worker-complete.json").is_file()
    runner.run_evaluation(receipt)
    assert len(calls) == 2  # Successful retry is reused without another model/runtime load.
    marker = output / "native/worker-complete.json"
    saved_marker = marker.read_bytes()
    marker.unlink()
    with pytest.raises(ValueError, match="no worker completion marker"):
        runner.run_evaluation(receipt)
    marker.write_bytes(saved_marker)
    changed = copy.deepcopy(receipt)
    changed["evaluation"]["worker"]["profile"]["context_length"] = 32768
    with pytest.raises(ValueError, match="different inputs"):
        runner.run_evaluation(changed)


def test_serialized_worker_runs_without_application_checkout(receipt, tmp_path):
    """The pinned image needs only stdlib and these two carried source files."""
    script = evaluation.specification(receipt)["tasks"][0]["arguments"][0]
    setup = shlex.split(script.splitlines()[1])[2]
    for name in ("evaluate.py", "evaluation.json", "bfcl_runtime.py"):
        setup = setup.replace("/tmp/" + name, str(tmp_path / name))
    probe = """
import json, pathlib, runpy, shlex
namespace = runpy.run_path(str(pathlib.Path.cwd() / 'evaluate.py'))
receipt = json.loads(pathlib.Path('evaluation.json').read_text())
worker = namespace['native_bfcl_worker'](receipt)
command = worker.launch_command(receipt, pathlib.Path('/weka/test/native'))
argv = shlex.split(command.splitlines()[-1])
assert argv[argv.index('--model') + 1] == receipt['checkpoint']
assert argv.count('--task') == 13
assert 'open_instruct' not in command
assert '--resume' in argv
"""
    subprocess.run([sys.executable, "-I", "-c", setup + "\n" + probe], cwd=tmp_path, check=True)
