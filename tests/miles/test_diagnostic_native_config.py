"""Fail-closed guards for isolated full-model native dispatch experiments."""

import hashlib
import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import diagnostic_native_config as diagnostic
from scripts.miles import qualify_math_weight_publication as gate


class Tuner:
    def __init__(self):
        self.arg_names = ["x", "N"]
        self.keys = ["N"]
        self.configs = [SimpleNamespace(kwargs={"BH": n}, num_warps=4) for n in (16, 32)]
        self.cache = {}
        self.best_config = None

    def run(self, *args, **kwargs):
        key = diagnostic.runtime_key(self, args, kwargs)
        self.best_config = self.cache.setdefault(key, self.configs[0])
        return args[0]


def setup(monkeypatch, tmp_path):
    tuner = Tuner()
    # Match the real exported kernel's heuristic wrapper.
    module = SimpleNamespace(chunk_kda_fwd_kernel_intra_token_parallel=SimpleNamespace(fn=tuner))
    monkeypatch.setattr(diagnostic.importlib, "import_module", lambda _: module)
    receipt = {
        "diagnostic_only": True,
        "target": diagnostic.TARGET,
        "autotuner_run_sha256": hashlib.sha256(inspect.getsource(Tuner.run).encode()).hexdigest(),
        "cache": {"(128, 'torch.bfloat16')": {"kwargs": {"BH": 32}, "num_warps": 4}},
    }
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    writes = {}
    return tuner, receipt, path, lambda p, v: writes.update({p.name: dict(v)}), writes


def test_native_key_selection_and_actual_launch_receipts(monkeypatch, tmp_path):
    tuner, _, path, save, writes = setup(monkeypatch, tmp_path)
    other = (64, "torch.bfloat16")
    tuner.cache[other] = tuner.configs[0]
    diagnostic.install(path, tmp_path, save)
    tensor = torch.zeros(1, dtype=torch.bfloat16)
    for _ in range(2):
        assert tuner.run(tensor, N=128) is tensor
        assert tuner.best_config is tuner.configs[1]
    assert tuner.run(tensor, 64) is tensor
    assert tuner.cache[other] is tuner.configs[0]
    assert writes["native-intervention-post.json"]["matched_calls"] == 2
    assert writes["native-intervention-post.json"]["other_key_calls"] == 1
    assert writes["native-intervention-pre.json"]["actual_key"] == "(128, 'torch.bfloat16')"
    assert not writes["native-intervention-installed.json"]["qualification_passed"]


@pytest.mark.parametrize("failure", ["source", "config", "target", "multiple", "diagnostic"])
def test_invalid_intervention_fails_before_mutating_native_cache(monkeypatch, tmp_path, failure):
    tuner, receipt, path, save, _ = setup(monkeypatch, tmp_path)
    if failure == "source":
        receipt["autotuner_run_sha256"] = "changed"
    elif failure == "config":
        next(iter(receipt["cache"].values()))["kwargs"]["BH"] = 999
    elif failure == "target":
        receipt["target"] = "another.kernel"
    elif failure == "multiple":
        receipt["cache"]["(64,)"] = next(iter(receipt["cache"].values()))
    else:
        receipt["diagnostic_only"] = False
    path.write_text(json.dumps(receipt))
    original = tuner.run
    with pytest.raises(ValueError):
        diagnostic.install(path, tmp_path, save)
    assert tuner.run == original and tuner.cache == {}


def test_strict_refresh_gate_refuses_intervention_before_cuda_or_artifacts(monkeypatch):
    monkeypatch.setenv("OI_PUBLICATION_NATIVE_CONFIG_RECEIPT", "diagnostic.json")
    with pytest.raises(ValueError, match="forbidden in strict"):
        gate.run(SimpleNamespace())


def test_two_recorded_tuners_are_independent_and_validated_before_install(monkeypatch, tmp_path):
    tuner, receipt, path, _, _ = setup(monkeypatch, tmp_path)
    gate_tuner = Tuner()
    modules = {
        diagnostic.TARGET.rsplit(".", 2)[0]: SimpleNamespace(
            chunk_kda_fwd_kernel_intra_token_parallel=SimpleNamespace(fn=tuner)
        ),
        diagnostic.GATE_TARGET.rsplit(".", 2)[0]: SimpleNamespace(
            kda_gate_chunk_cumsum_vector_kernel=SimpleNamespace(fn=gate_tuner)
        ),
    }
    monkeypatch.setattr(diagnostic.importlib, "import_module", modules.__getitem__)
    receipt["additional_interventions"] = [{**receipt, "target": diagnostic.GATE_TARGET}]
    receipt["additional_interventions"][0]["autotuner_run_sha256"] = "bad"
    path.write_text(json.dumps(receipt))
    original = tuner.run
    with pytest.raises(ValueError, match="implementation changed"):
        diagnostic.install(path, tmp_path, lambda *_: None)
    assert tuner.run == original and tuner.cache == {} and gate_tuner.cache == {}
    receipt["additional_interventions"][0]["autotuner_run_sha256"] = receipt["autotuner_run_sha256"]
    path.write_text(json.dumps(receipt))
    writes = {}
    diagnostic.install(path, tmp_path, lambda p, v: writes.update({str(p): dict(v)}))
    tensor = torch.zeros(1, dtype=torch.bfloat16)
    for _ in range(2):
        for selected in (tuner, gate_tuner):
            assert selected.run(tensor, 128) is tensor
            assert selected.best_config is selected.configs[1]
    primary = writes[str(tmp_path / "native-intervention-post.json")]
    extra = writes[str(tmp_path / "gate-native-intervention/native-intervention-post.json")]
    assert primary["target"] == diagnostic.TARGET and extra["target"] == diagnostic.GATE_TARGET
    assert primary["matched_calls"] == extra["matched_calls"] == 2


@pytest.mark.parametrize("extra", [[{"target": "another.kernel"}], [{}, {}]])
def test_unapproved_additional_tuners_fail_closed(monkeypatch, tmp_path, extra):
    tuner, receipt, path, save, _ = setup(monkeypatch, tmp_path)
    receipt["additional_interventions"] = extra
    path.write_text(json.dumps(receipt))
    original = tuner.run
    with pytest.raises(ValueError, match="residual gate"):
        diagnostic.install(path, tmp_path, save)
    assert tuner.run == original and tuner.cache == {}
