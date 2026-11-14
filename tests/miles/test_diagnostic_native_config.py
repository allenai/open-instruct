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
