"""Offline validation of diagnostic replay's native-config identity contract."""

import json
import sys
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import diagnose_math_kda_dispatch as replay


def test_restore_dispatch_uses_runtime_config_objects_and_unwraps(monkeypatch):
    config = SimpleNamespace(kwargs={"BV": 32}, num_warps=2, num_stages=3)
    tuner = SimpleNamespace(configs=[config], cache={"stale": None})
    module = SimpleNamespace(kernel=SimpleNamespace(fn=tuner))
    monkeypatch.setattr(replay.importlib, "import_module", lambda name: module if name == "fla.example" else None)
    count = replay.restore_dispatch(
        {
            "fla.example.kernel.fn": {
                "cache": {"(8, 'torch.bfloat16')": {"kwargs": {"BV": 32}, "num_warps": 2, "num_stages": 3}}
            }
        }
    )
    assert count == {"fla.example.kernel.fn": 1}
    assert tuner.cache == {(8, "torch.bfloat16"): config}
    assert next(iter(tuner.cache.values())) is config


def test_restore_dispatch_rejects_config_absent_from_pinned_runtime(monkeypatch):
    tuner = SimpleNamespace(configs=[SimpleNamespace(kwargs={"BV": 32})], cache={})
    monkeypatch.setattr(replay.importlib, "import_module", lambda name: SimpleNamespace(kernel=tuner))
    with pytest.raises(ValueError, match="not uniquely present"):
        replay.restore_dispatch({"fla.example.kernel": {"cache": {"(8,)": {"kwargs": {"BV": 128}}}}})
    assert tuner.cache == {}


def test_output_comparison_preserves_actual_differences_and_finiteness():
    a = torch.tensor([0, 1], dtype=torch.bfloat16)
    b = torch.tensor([0, 1.5], dtype=torch.bfloat16)
    assert replay.difference(a, b) == {
        "equal": False,
        "finite": True,
        "max_abs": 0.5,
        "changed_elements": 1,
        "elements": 2,
    }
    assert not replay.difference(a, b * float("nan"))["finite"]
    with pytest.raises(ValueError, match="layout"):
        replay.difference(a, b.float())


def test_single_config_interventions_hold_other_recorded_choices_fixed():
    base = {"first": {"cache": {"(1,)": "a"}}, "second": {"cache": {"(2,)": "b"}}}
    target = {"first": {"cache": {"(1,)": "c"}}, "second": {"cache": {"(2,)": "d"}}}
    changes = list(replay.single_config_changes(base, target))
    assert [name for name, _ in changes] == ["first", "second"]
    for name, snapshot in changes:
        assert snapshot[name] == target[name]
        other = "second" if name == "first" else "first"
        assert snapshot[other] == base[other]
    changes[0][1]["second"]["cache"].clear()
    assert base["second"]["cache"] == {"(2,)": "b"}
    assert target["second"]["cache"] == {"(2,)": "d"}
    with pytest.raises(ValueError, match="sets differ"):
        list(replay.single_config_changes(base, {}))


def test_capture_inspection_rejects_state_and_semantic_confounds(tmp_path):
    records = []
    captured = []
    for index in range(3):
        inputs = tmp_path / f"inputs{index}.pt"
        output = tmp_path / f"output{index}.pt"
        data = {"initial_state": torch.zeros(2), "options": {"scale": 1.0}, "positional": [torch.ones(2)]}
        torch.save(data, inputs)
        torch.save(torch.tensor([float(index)]), output)
        receipt = {"prefill_recurrence_index": 1, "core_compat": True, "allow_neg_eigval": False, "fla_dispatch": {}}
        records.append((f"gpu{index + 1}", receipt, inputs, output))
        captured.append(data)
    report = replay.inspect_captures(records, captured, expected_index=1)
    assert report["causal_replay_allowed"]
    assert not report["qualification_passed"]
    assert len(report["pairs"]) == 3
    assert report["pairs"][0]["output_comparison"]["max_abs"] == 1.0
    captured[1]["initial_state"][0] = 1
    captured[2]["options"]["scale"] = 2.0
    records[2][1]["core_compat"] = False
    report = replay.inspect_captures(records, captured, expected_index=1)
    assert not report["causal_replay_allowed"]
    assert report["pairs"][0]["unequal_input_paths"] == ["inputs.initial_state"]
    assert report["pairs"][1]["unequal_input_paths"] == ["inputs.options.scale"]
    assert report["pairs"][1]["unequal_semantics"] == ["core_compat"]
    with pytest.raises(ValueError, match="recurrence index"):
        replay.inspect_captures(records, captured, expected_index=0)
    del records[1][1]["prefill_recurrence_index"]
    with pytest.raises(ValueError, match="recurrence index"):
        replay.inspect_captures(records, captured, expected_index=1)
    assert not replay.inspect_captures(records, captured)["causal_replay_allowed"]


def test_capture_input_identity_includes_dtype_shape_and_nested_options():
    assert replay.unequal_paths({"x": torch.ones(2)}, {"x": torch.ones(2).bfloat16()}) == ["inputs.x"]
    assert replay.unequal_paths((1, [2]), (1, [3])) == ["inputs[1][0]"]
    assert replay.unequal_paths({"x": 1}, {"y": 1}) == ["inputs"]


@pytest.mark.parametrize("nested", [False, True])
def test_inspect_only_cli_records_confounds_without_initializing_cuda(tmp_path, monkeypatch, nested):
    for index in range(3):
        directory = tmp_path / "captures" / f"gpu{index + 1}"
        if nested:
            directory = directory / "recurrence-1"
        directory.mkdir(parents=True)
        torch.save({"initial_state": torch.tensor([float(index)])}, directory / "kda-inputs-1.pt")
        torch.save(torch.tensor([float(index)]), directory / "kda-output-1.pt")
        (directory / "kda-receipt-1.json").write_text(
            json.dumps(
                {"prefill_recurrence_index": 1, "core_compat": False, "allow_neg_eigval": True, "fla_dispatch": {}}
            )
        )
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("CPU inspection must not initialize CUDA"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnostic",
            "--capture",
            str(tmp_path / "captures"),
            "--output",
            str(tmp_path / "inspection"),
            "--inspect-only",
            "--expected-recurrence-index",
            "1",
        ],
    )
    replay.main()
    report = json.loads((tmp_path / "inspection/capture-inspection.json").read_text())
    assert not report["causal_replay_allowed"]
    assert len(report["pairs"]) == 3
    assert all(pair["unequal_input_paths"] == ["inputs.initial_state"] for pair in report["pairs"])


def test_explicit_receiver_subset_excludes_confounded_receiver(tmp_path, monkeypatch):
    for index in (1, 2, 3):
        directory = tmp_path / "captures" / f"gpu{index}"
        directory.mkdir(parents=True)
        torch.save({"initial_state": torch.tensor([float(index == 1)])}, directory / "kda-inputs-1.pt")
        torch.save(torch.tensor([float(index)]), directory / "kda-output-1.pt")
        (directory / "kda-receipt-1.json").write_text(
            json.dumps(
                {"prefill_recurrence_index": 2, "core_compat": True, "allow_neg_eigval": False, "fla_dispatch": {}}
            )
        )
    argv = [
        "diagnostic",
        "--capture",
        str(tmp_path / "captures"),
        "--output",
        str(tmp_path / "inspection"),
        "--inspect-only",
        "--expected-recurrence-index",
        "2",
        "--receivers",
        "gpu2",
        "gpu3",
    ]
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("CPU inspection must not initialize CUDA"))
    monkeypatch.setattr(sys, "argv", argv)
    replay.main()
    report = json.loads((tmp_path / "inspection/capture-inspection.json").read_text())
    assert report["causal_replay_allowed"]
    assert report["pairs"][0]["receivers"] == ["gpu2", "gpu3"]
    assert report["pairs"][0]["output_comparison"]["max_abs"] == 1.0
    for receivers, message in [(["gpu2", "gpu2"], "distinct"), (["gpu2", "gpu4"], "missing")]:
        attempt = [*argv[:-2], *receivers]
        attempt[attempt.index("--output") + 1] = str(tmp_path / message)
        monkeypatch.setattr(sys, "argv", attempt)
        with pytest.raises(ValueError, match=message):
            replay.main()
