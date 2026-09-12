"""CPU launch contracts; subprocesses are mocked except shell syntax checks."""

import ast
import base64
import json
import shlex
import subprocess
from pathlib import Path

import pytest
from scripts.miles import launch_workflow

from open_instruct.miles import launch, workflow
from open_instruct.miles.run_spec import RunSpec

IMAGE_ID = "01M2931KARFP3Y2W2FPADGRFEP"


def spec(tmp_path, **sections):
    document = {
        "schema_version": 1,
        "name": "researcher-trial",
        "model": {"source": "/weka/oe-training-default/model with 'quotes' $(touch should-not-exist)"},
        "output": {"root": "/weka/oe-training-default/run with spaces"},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 32, "eval_count": 16}]},
    }
    return RunSpec.from_dict(document | sections, config_path=tmp_path / "run.toml")


def payload(command):
    setup = next(line for line in command.splitlines() if line.startswith("python -c "))
    source = shlex.split(setup)[2]
    tree = ast.parse(source)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "b64decode"
    )
    return json.loads(base64.b64decode(ast.literal_eval(call.args[0])))


def test_payload_is_exact_resolved_spec_with_quoted_paths_and_overrides(tmp_path):
    original = spec(tmp_path)
    run = RunSpec.from_dict(
        original.to_dict(),
        config_path=tmp_path / "other" / "config.json",
        overrides=["optimizer.learning_rate=0.000003", 'tracking.wandb_run_name="literal `x` $HOME \\"q\\""'],
    )
    task = launch.specification(IMAGE_ID, run)["tasks"][0]
    command = task["arguments"][0]
    assert payload(command) == run.to_dict()
    assert payload(command)["optimizer"]["learning_rate"] == 3e-6
    assert task["image"] == {"beaker": IMAGE_ID}
    subprocess.run(["bash", "-n"], input=command, text=True, check=True)
    assert not (tmp_path / "should-not-exist").exists()


@pytest.mark.parametrize("placement,serving,gpus", [("colocated", 2, 2), ("disaggregated", 1, 3)])
def test_single_node_gpu_accounting(tmp_path, placement, serving, gpus):
    run = spec(tmp_path, inference={"placement_mode": placement, "gpus": serving})
    task = launch.specification(IMAGE_ID, run)["tasks"][0]
    assert task["resources"]["gpuCount"] == gpus
    assert task["constraints"]["cluster"] == ["ai2/holmes"]
    assert task["context"]["priority"] == "urgent"
    assert task["context"]["minRuntime"] == "1h"


def test_multinode_auto_resume_requires_qualification(tmp_path):
    run = spec(tmp_path, inference={"placement_mode": "disaggregated", "gpus": 2}, launch={"gpus_per_replica": 3})
    with pytest.raises(ValueError, match="auto_resume=false"):
        launch.specification(IMAGE_ID, run)


@pytest.mark.parametrize("section", ["data", "conversion", "compiler_cache", "miles"])
def test_every_weka_input_and_output_requires_mount_coverage(tmp_path, section):
    value = {
        "data": {"rl_manifest": "/weka/other-data/task/rl-manifest.json"},
        "conversion": {"hf_output": "/weka/other-data/hf"},
        "compiler_cache": {"enabled": True, "shared_root": "/weka/other-data/tmp-30d/cache"},
        "miles": {"load": "/weka/other-data/checkpoints"},
    }[section]
    run = spec(tmp_path, **{section: value})
    with pytest.raises(ValueError, match="weka_mounts"):
        launch.specification(IMAGE_ID, run)
    run.launch["weka_mounts"].append({"weka": "other-data", "mount_path": "/weka/other-data"})
    assert len(launch.specification(IMAGE_ID, run)["tasks"][0]["datasets"]) == 2


def test_credentials_are_secret_references_and_never_inherited_plaintext(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "PRIVATE-VALUE-SENTINEL")
    run = spec(tmp_path, launch={"secrets": {"HF_TOKEN": "robertb_hf_secret"}, "env": {"DEBUG_LABEL": "visible"}})
    document = launch.specification(IMAGE_ID, run)
    env = document["tasks"][0]["envVars"]
    assert {"name": "HF_TOKEN", "secret": "robertb_hf_secret"} in env
    assert not any(value["name"] == "HF_TOKEN" and "value" in value for value in env)
    assert "PRIVATE-VALUE-SENTINEL" not in json.dumps(document)
    run.launch["env"]["WANDB_API_KEY"] = "should-not-print"
    with pytest.raises(ValueError, match="launch.secrets") as error:
        launch.specification(IMAGE_ID, run)
    assert "should-not-print" not in str(error.value)


def test_run_freezes_overrides_before_required_build_wrapper(tmp_path, monkeypatch):
    path = tmp_path / "run.toml"
    path.write_text(
        'schema_version=1\nname="freeze-test"\n[model]\nsource="/weka/oe-training-default/hf"\n[output]\nroot="/weka/oe-training-default/run"\n[data]\n[[data.tasks]]\ntask="multiplication"\ntrain_count=8\n'
    )
    expected = RunSpec.load(path, ["optimizer.learning_rate=0.000004"]).to_dict()
    calls = []

    def run(command, **kwargs):
        assert command[:4] == [
            "bash",
            "./scripts/train/build_image_and_launch.sh",
            "--miles",
            "scripts/train/debug/miles_workflow.sh",
        ]
        frozen = Path(command[4])
        assert frozen != path and frozen.suffix == ".json"
        assert json.loads(frozen.read_text()) == expected
        path.write_text("changed while image builds")
        assert json.loads(frozen.read_text()) == expected
        assert kwargs == {"cwd": launch.ROOT, "check": True}
        calls.append(command)

    monkeypatch.setattr(launch.subprocess, "run", run)
    launch.run(path, ["optimizer.learning_rate=0.000004"])
    assert len(calls) == 1 and not Path(calls[0][4]).exists()


def test_submit_resolves_image_and_records_exact_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("MILES_LAUNCH_RECEIPTS", str(tmp_path / "receipts"))
    run = spec(tmp_path)
    submitted = []

    def check_output(command, **kwargs):
        if command[:3] == ["beaker", "image", "get"]:
            assert command[3] == "robertb/image-alias"
            return json.dumps([{"id": IMAGE_ID}])
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            assert kwargs["cwd"] == launch.ROOT
            return "source-revision\n"
        assert command[:3] == ["beaker", "experiment", "create"]
        assert command[4:] == ["--workspace", "ai2/open-instruct-dev", "--format", "json"]
        submitted.append(json.loads(Path(command[3]).read_text()))
        return json.dumps([{"id": f"experiment-{len(submitted)}"}])

    monkeypatch.setattr(launch.subprocess, "check_output", check_output)
    first = launch.submit("robertb/image-alias", run)
    assert first["image"] == IMAGE_ID and first["requested_image"] == "robertb/image-alias"
    assert first["revision"] == "source-revision"
    assert first["spec_sha256"] == workflow.fingerprint(run.to_dict())
    assert payload(submitted[0]["tasks"][0]["arguments"][0]) == first["spec"]
    assert submitted[0]["tasks"][0]["image"]["beaker"] == IMAGE_ID
    second = launch.submit("robertb/image-alias", run)
    assert json.loads(launch.receipt_path(run).read_text()) == second
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 2


def test_status_uses_actual_beaker_top_level_job_shape_and_latest_attempt(tmp_path, monkeypatch):
    # Shape and timestamp fields verified against retained replay-reaudit
    # experiment 01M295HY6X4NMBGB8R0JW24EBQ (no live Beaker request).
    monkeypatch.setenv("MILES_LAUNCH_RECEIPTS", str(tmp_path / "receipts"))
    run = spec(tmp_path)
    workflow.write_json(
        launch.receipt_path(run), {"experiment_id": "experiment", "spec_sha256": workflow.fingerprint(run.to_dict())}
    )
    experiment = {
        "id": "experiment",
        "jobs": [
            {
                "kind": "execution",
                "id": "new",
                "status": {
                    "created": "2026-09-11T21:21:39.549634Z",
                    "started": "2026-09-11T21:22:06.583836Z",
                    "exitCode": 0,
                },
            },
            {"kind": "execution", "id": "old", "status": {"created": "2026-09-11T20:00:00Z", "exitCode": 143}},
        ],
    }
    monkeypatch.setattr(launch.subprocess, "check_output", lambda *args, **kwargs: json.dumps([experiment]))
    result = launch.status(run)
    assert result["latest_job"]["id"] == "new"
    assert [job["id"] for job in result["attempts"]] == ["old", "new"]
    assert result["config_matches_submission"] is True
    run.data["seed"] += 1
    assert launch.status(run)["config_matches_submission"] is False


def test_result_collection_reads_only_small_reports_and_prunes_weight_trees(tmp_path, monkeypatch):
    root, destination = tmp_path / "run", tmp_path / "result"
    (root / "checkpoints/step1/model").mkdir(parents=True)
    (root / "prepared/hf").mkdir(parents=True)
    (root / "metrics").mkdir()
    (root / "workflow.json").write_text("{}")
    (root / "checkpoints/step1/complete.json").write_text("{}")
    (root / "checkpoints/step1/model/large.json").write_text("must not read")
    (root / "prepared/hf/config.json").write_text("must not read")
    large = root / "metrics/too-large.jsonl"
    with large.open("wb") as stream:
        stream.truncate(33 * 1024 * 1024)
    (root / "metrics/train.jsonl").write_text('{"step":1}\n')
    (root / "metrics/link.json").symlink_to(root / "prepared/hf/config.json")
    original = Path.read_bytes

    def read(path):
        assert path not in (large, root / "prepared/hf/config.json", root / "checkpoints/step1/model/large.json")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    result = launch.collect_results(root, destination)
    assert set(result) == {"workflow.json", "checkpoints/step1/complete.json", "metrics/train.jsonl"}
    with pytest.raises(ValueError, match="outside"):
        launch.collect_results(root, root / "recursive")


def test_submission_script_accepts_frozen_json(tmp_path, monkeypatch):
    run = spec(tmp_path)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(run.to_dict()))
    calls = []
    monkeypatch.setattr(
        launch_workflow.launch, "submit", lambda image, actual: calls.append((image, actual.to_dict()))
    )
    monkeypatch.setattr(
        launch_workflow.argparse.ArgumentParser,
        "parse_args",
        lambda parser: type("Args", (), {"image": IMAGE_ID, "config": path, "overrides": [], "render_only": False})(),
    )
    launch_workflow.main()
    assert calls == [(IMAGE_ID, run.to_dict())]
