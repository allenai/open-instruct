"""Profile selection and capacity-aware batching at the evaluator boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError
from open_instruct.miles.evaluation import evaluation_runner

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def document():
    return RunSpec.load(ROOT / "configs/miles/examples/three-node.toml").to_dict()


@pytest.mark.parametrize("example", ["node", "three-node"])
def test_examples_require_qualified_profile_and_capacity_batches(example):
    config = RunSpec.load(ROOT / f"configs/miles/examples/{example}.toml").evaluation
    assert config["provider_kwargs"] == {"profile": "olmo3.5-12.5b", "context_length": 34816}
    assert config["external_server_args"] == []
    for task in config["tasks"]:
        args = evaluation_runner.command({"checkpoint": "/hf", "evaluation": config, "tasks": [task]}, "/out")
        if task.get("kind") != "external":
            assert "provider.kwargs.profile=olmo3.5-12.5b" in args
            assert "batching.chunk_size=auto" in args
            assert not any("disable_cuda_graph" in arg or "disable_radix_cache" in arg for arg in args)
            assert not any(arg.startswith("batching.max_batch_sequences=") for arg in args)


def test_default_profile_is_frozen_in_receipt_and_explicit_overrides_win(document):
    document["evaluation"]["provider_kwargs"] = {"context_length": 34816}
    config = RunSpec.from_dict(document).evaluation
    assert config["provider_kwargs"]["profile"] == "auto"
    config["provider_kwargs"] = {"profile": "generic", "disable_cuda_graph": True}
    task = {**config["tasks"][0], "harness": {"batching.chunk_size": 16}}
    args = evaluation_runner.command({"checkpoint": "/hf", "evaluation": config, "tasks": [task]}, "/out")
    overrides = dict(arg.split("=", 1) for i, arg in enumerate(args) if args[i - 1] == "-o")
    assert overrides["provider.kwargs.profile"] == "generic"
    assert overrides["provider.kwargs.disable_cuda_graph"] == "true"
    assert overrides["batching.chunk_size"] == "16"


@pytest.mark.parametrize("profile", ["auto", "olmo3.5-12.5b"])
def test_scicode_rejects_separate_cli_settings_that_bypass_profile(document, profile):
    document["evaluation"]["provider_kwargs"] = {"profile": profile}
    document["evaluation"]["external_server_args"] = ["--disable-cuda-graph"]
    with pytest.raises(InputError, match="Profiled SciCode"):
        RunSpec.from_dict(document)
    document["evaluation"]["provider_kwargs"]["profile"] = "generic"
    assert RunSpec.from_dict(document).evaluation["external_server_args"] == ["--disable-cuda-graph"]


@pytest.mark.parametrize("failure", [None, "profile", "server"])
def test_scicode_resolves_profile_before_starting_server_and_records_it(tmp_path, monkeypatch, failure):
    options = {"context_length": 34816, "cuda_graph_backend_decode": "full", "disable_radix_cache": False}
    report = {"selected": "olmo3.5-12.5b-sglang-b300-v1", "defaults": options}
    prepare = Mock(return_value=report)
    start = Mock()
    cleanup = Mock()
    if failure == "profile":
        prepare.side_effect = ValueError("not qualified")
    if failure == "server":
        start.side_effect = RuntimeError("startup failed")
    modules = {
        "olmo_eval.inference.providers.sglang_profiles": SimpleNamespace(prepare_profile=prepare),
        "olmo_sglang": SimpleNamespace(register=Mock()),
        "sglang.srt.plugins": SimpleNamespace(load_plugins=Mock()),
        "sglang.srt.server_args": SimpleNamespace(ServerArgs=lambda **kwargs: kwargs),
        "sglang.launch_server": SimpleNamespace(run_server=start),
        "sglang.srt.utils": SimpleNamespace(kill_process_tree=cleanup),
    }
    monkeypatch.setattr(evaluation_runner.importlib, "import_module", modules.__getitem__)
    receipt = {
        "checkpoint": "/frozen/hf",
        "evaluation": {"gpus": 1, "provider_kwargs": {"profile": "olmo3.5-12.5b", "max_model_len": 34816}},
    }
    if failure:
        with pytest.raises(ValueError if failure == "profile" else RuntimeError):
            evaluation_runner.serve(receipt, tmp_path)
    else:
        evaluation_runner.serve(receipt, tmp_path)
    selector, supplied, registration = prepare.call_args.args
    assert selector == "olmo3.5-12.5b"
    assert registration == "olmo_sglang:register"
    assert supplied["tp_size"] == 1 and supplied["model_path"] == "/frozen/hf"
    assert supplied["context_length"] == 34816 and "max_model_len" not in supplied
    if failure == "profile":
        start.assert_not_called()
        cleanup.assert_not_called()
    else:
        effective = start.call_args.args[0]
        assert effective["cuda_graph_backend_decode"] == "full"
        assert effective["disable_radix_cache"] is False
        assert effective["host"] == "127.0.0.1" and effective["port"] == 30000
        saved = json.loads((tmp_path / "sglang-external-profile.json").read_text())
        assert saved["selected"] == report["selected"]
        assert saved["effective_options"] == effective
        cleanup.assert_called_once()
