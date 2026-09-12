"""Invalid researcher inputs fail on CPU with field context and a usable correction."""

import json
import sys
from types import SimpleNamespace

import pytest

from open_instruct.miles import __main__ as cli
from open_instruct.miles import options, run_data, workflow
from open_instruct.miles.config import CoreConfig, RunConfig
from open_instruct.miles.errors import InputError
from open_instruct.miles.run_spec import RunSpec


def raw_config(**values):
    return {"miles": {"hf_checkpoint": "model", "global_batch_size": 8, **values}}


@pytest.mark.parametrize("value", [True, "0.1", [], float("nan"), float("inf"), -0.1])
@pytest.mark.parametrize(
    "field",
    [
        "router_aux_loss_weight",
        "router_z_loss_weight",
        "max_train_rollout_logprob_abs_diff",
        "scoring_check_tolerance",
    ],
)
def test_core_numbers_reject_invalid_values_without_type_errors(field, value):
    with pytest.raises(InputError, match=rf"core\.{field}.*finite number"):
        CoreConfig(**{field: value})


@pytest.mark.parametrize(
    "field,value",
    [
        ("lr", -1),
        ("adam_eps", 0),
        ("adam_beta1", 1),
        ("adam_beta2", -0.1),
        ("sglang_mem_fraction_static", 0),
        ("sglang_mem_fraction_static", 1.1),
        ("rollout_top_p", 0),
        ("eval_temperature", -1),
        ("rollout_max_context_len", 0),
        ("eval_interval", 0),
        ("save_interval", -1),
        ("global_batch_size", True),
        ("async_data_buffer_capacity_factor", 0),
        ("lr_warmup_fraction", 1),
    ],
)
def test_native_numeric_ranges_fail_before_runtime_import(field, value):
    with pytest.raises(InputError, match=rf"miles\.{field}"):
        RunConfig.from_dict(raw_config(**{field: value}))


def test_valid_boundaries_and_native_sentinels_remain_supported():
    values = dict(
        lr=0,
        min_lr=0,
        lr_warmup_init=0,
        weight_decay=0,
        clip_grad=0,
        adam_beta1=0,
        adam_beta2=0,
        rollout_temperature=0,
        eval_temperature=0,
        rollout_top_p=1,
        eval_top_p=1,
        sglang_mem_fraction_static=1,
        rollout_top_k=-1,
        eval_top_k=-1,
        seed=0,
    )
    config = RunConfig.from_dict(raw_config(**values))
    for field, value in values.items():
        assert config.miles[field] == value
    assert "--eval-temperature" in config.arguments()


@pytest.mark.parametrize(
    "values,field",
    [
        ({"lr": 1e-6, "min_lr": 1e-5}, "min_lr"),
        ({"lr": 1e-6, "lr_warmup_init": 1e-5}, "lr_warmup_init"),
        ({"lr_decay_iters": 10, "lr_warmup_iters": 10}, "lr_warmup_iters"),
    ],
)
def test_scheduler_bounds_have_field_context(values, field):
    with pytest.raises(InputError, match=rf"miles\.{field}"):
        RunConfig.from_dict(raw_config(**values))


def test_fractional_warmup_preserves_scheduler_precedence():
    RunConfig.from_dict(raw_config(lr_decay_iters=10, lr_warmup_iters=20, lr_warmup_fraction=0.1))


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"core": []}, r"\[core\].*table/object"),
        ({"core": {"expert_paralel_size": 2}}, "did you mean 'expert_parallel_size'"),
        ({"miles": {1: "value"}}, "named fields"),
        ({"core": {"compiler_cache": "false"}}, "true or false without quotes"),
    ],
)
def test_sections_and_typo_messages(payload, match):
    with pytest.raises(InputError, match=match):
        RunConfig.from_dict(payload)


@pytest.mark.parametrize(
    "override,match",
    [
        ("core.attention_backend=flash_4", "quote strings"),
        ("core.expert_parallel_size", "core.KEY=TOML_VALUE"),
        (None, "Override.*nonempty string"),
    ],
)
def test_overrides_are_actionable(override, match):
    with pytest.raises(InputError, match=match):
        RunConfig.from_dict(raw_config(), [override])


def test_topology_message_explains_actual_counts():
    payload = raw_config(actor_num_gpus_per_node=3)
    payload["core"] = {"expert_parallel_size": 2}
    with pytest.raises(InputError, match=r"world size 3.*expert_parallel_size=2.*divisor of 3"):
        RunConfig.from_dict(payload)


def test_researcher_alias_keeps_original_field_context(tmp_path):
    payload = {
        "schema_version": 1,
        "name": "trial",
        "model": {"source": "model"},
        "output": {"root": str(tmp_path / "run")},
        "data": {"tasks": [{"task": "gsm8k", "train_count": 8}]},
        "optimizer": {"learning_rate": -1},
    }
    with pytest.raises(InputError, match=r"optimizer.learning_rate: miles.lr"):
        RunSpec.from_dict(payload)


def test_json_option_error_identifies_the_option():
    with pytest.raises(InputError, match="miles.sglang_json_model_override_args.*JSON"):
        options.encode_options({"sglang_json_model_override_args": '{"invalid":'})


@pytest.mark.parametrize(
    "raw,match",
    [
        (b'{}\n\n{"bad":\n', "questions.jsonl: line 3.*JSON"),
        (b"{}\n[]\n", "questions.jsonl: line 2:.*objects"),
        (b"\xff", "questions.jsonl.*UTF-8"),
    ],
)
def test_jsonl_errors_identify_file_and_line(raw, match):
    with pytest.raises(InputError, match=match):
        run_data._rows(raw, "questions.jsonl")


def test_malformed_verifier_does_not_raise_type_error_or_echo_target():
    row = {"input": "question", "metadata": {"verifiers": [{"name": [], "target": "private target"}]}}
    with pytest.raises(InputError, match=r"metadata.verifiers\[0\].*gsm8k") as error:
        run_data._verify_row(row, None, 1024, {"gsm8k"})
    assert "private target" not in str(error.value)


@pytest.mark.parametrize("content", ["[core]\ncompiler_cache='false'", "[core", "[]", None])
def test_cli_input_errors_exit_two_without_runtime_import(tmp_path, monkeypatch, capsys, content):
    path = tmp_path / ("run.json" if content == "[]" else "run.toml")
    if content is not None:
        path.write_text(content)
    monkeypatch.setattr(sys, "argv", ["miles", "train", str(path)])
    monkeypatch.setattr(cli.importlib, "import_module", lambda name: pytest.fail(f"Imported runtime: {name}"))
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert str(path) in stderr and "error:" in stderr and "Traceback" not in stderr


def test_debug_preserves_input_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["miles", "plan", str(tmp_path / "missing.toml"), "--debug"])
    with pytest.raises(InputError, match="Cannot read configuration"):
        cli.main()


def test_unexpected_failure_is_not_mislabeled_as_input(tmp_path, monkeypatch):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(raw_config()))
    monkeypatch.setattr(sys, "argv", ["miles", "train", str(path)])

    def broken(config):
        raise RuntimeError("unexpected backend defect")

    monkeypatch.setattr(cli.importlib, "import_module", lambda name: SimpleNamespace(train_config=broken))
    with pytest.raises(RuntimeError, match="unexpected backend defect"):
        cli.main()


def test_raw_json_plan_matches_toml(tmp_path, monkeypatch, capsys):
    json_path, toml_path = tmp_path / "run.json", tmp_path / "run.toml"
    json_path.write_text(json.dumps(raw_config()))
    toml_path.write_text('[miles]\nhf_checkpoint="model"\nglobal_batch_size=8\n')
    assert RunConfig.load(json_path).arguments() == RunConfig.load(toml_path).arguments()
    monkeypatch.setattr(sys, "argv", ["miles", "plan", str(json_path)])
    cli.main()
    assert json.loads(capsys.readouterr().out)["miles"]["global_batch_size"] == 8


def test_missing_model_points_to_source_and_mount(tmp_path):
    with pytest.raises(InputError, match=r"model.source.*launch.weka"):
        workflow.model_identity(tmp_path / "missing")


@pytest.mark.parametrize("content", [b"{", b"[]", b"null", b"\xff"])
def test_reward_registry_parse_errors_name_the_file(tmp_path, content):
    path = tmp_path / "rewards.json"
    path.write_bytes(content)
    with pytest.raises(InputError, match="rewards.json"):
        run_data._prepared({"reward_config": str(path)}, {})
