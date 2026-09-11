"""Pin only declared configurations and record actual armed invocations."""

import json
from types import SimpleNamespace

import pytest
import triton
from scripts.miles import update_zero_autotune as pin


class Tuner:
    def __init__(self, configs):
        self.configs = configs
        self.cache = {}
        self.calls = 0

    def run(self, value):
        self.calls += 1
        self.best_config = self.configs[0]
        return value + 1


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    monkeypatch.setattr(pin, "_STATE", {"profile": None, "capture": None})


def profile(tmp_path, configurations):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({"schema_version": 1, "configurations": configurations}))
    return path


def test_selects_actual_candidate_before_warmup_and_records_invocations(tmp_path):
    one, two = triton.Config({"BK": 32}, num_warps=4), triton.Config({"BK": 64}, num_warps=4)
    pinned, untouched = Tuner([one, two]), Tuner([one, two])
    path = profile(tmp_path, {"fla.kernel": pin.configuration(two)})
    result = pin.apply_reference(path, tuners={"fla.kernel": pinned, "fla.other": untouched})
    assert pinned.configs == [two] and pinned.configs[0] is two
    assert untouched.configs == [one, two]
    assert result["observed_unpinned_tuners"] == ["fla.other"]
    assert pinned.run(4) == 5  # Warmup is outside armed records.
    assert pin.finish_capture() is None
    pin.begin_capture("hf-case")
    pinned.run(1)
    untouched.run(2)
    recorded = pin.finish_capture()
    assert len(recorded["invocations"]) == 2
    assert recorded["invocations"][0] == {
        "kernel": "fla.kernel",
        "pinned": True,
        "configuration": pin.configuration(two),
    }
    assert not recorded["invocations"][1]["pinned"]
    assert recorded["pinned_tuners_without_observed_invocation"] == []
    assert pin.apply_reference(path, tuners={}) is result  # Idempotent same profile.


def test_invalid_second_candidate_does_not_partially_mutate_first(tmp_path):
    valid = triton.Config({"BK": 32})
    left, right = Tuner([valid]), Tuner([valid])
    path = profile(
        tmp_path, {"fla.left": pin.configuration(valid), "fla.right": pin.configuration(triton.Config({"BK": 999}))}
    )
    original = left.run
    with pytest.raises(ValueError, match="not a declared candidate"):
        pin.apply_reference(path, tuners={"fla.left": left, "fla.right": right})
    assert left.run == original and pin._STATE["profile"] is None


def test_missing_or_already_used_tuner_rejected(tmp_path):
    config = triton.Config({"BK": 32})
    path = profile(tmp_path, {"fla.kernel": pin.configuration(config)})
    with pytest.raises(ValueError, match="not loaded"):
        pin.apply_reference(path, tuners={})
    tuner = Tuner([config])
    tuner.cache["already"] = config
    with pytest.raises(ValueError, match="already executed"):
        pin.apply_reference(path, tuners={"fla.kernel": tuner})


def test_no_observed_invocations_is_explicit_and_capture_recovers(tmp_path):
    config = triton.Config({"BK": 32})
    pin.apply_reference(
        profile(tmp_path, {"fla.kernel": pin.configuration(config)}), tuners={"fla.kernel": Tuner([config])}
    )
    pin.begin_capture("one")
    with pytest.raises(ValueError, match="Nested"):
        pin.begin_capture("two")
    assert pin.finish_capture()["pinned_tuners_without_observed_invocation"] == ["fla.kernel"]
    pin.begin_capture("three")
    assert pin.finish_capture()["capture_id"] == "three"


def test_discovers_aliases_once_and_unwraps_decorators(monkeypatch):
    tuner = Tuner([])
    monkeypatch.setattr(pin, "Autotuner", Tuner)
    monkeypatch.setattr(pin.sys, "modules", {"fla.fake": SimpleNamespace(a=SimpleNamespace(fn=tuner), b=tuner)})
    assert pin.discover_tuners() == {"fla.fake.a": tuner}


def test_configuration_change_after_install_rejected(tmp_path):
    config = triton.Config({"BK": 32})
    path = profile(tmp_path, {"fla.kernel": pin.configuration(config)})
    pin.apply_reference(path, tuners={"fla.kernel": Tuner([config])})
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="Cannot replace"):
        pin.apply_reference(path, tuners={})
