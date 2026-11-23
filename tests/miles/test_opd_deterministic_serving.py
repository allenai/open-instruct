"""Exercise serving-option isolation and the submitted allocation contract."""

from pathlib import Path

import pytest

from open_instruct.miles.configuration import specs
from open_instruct.miles.distillation import opd_launch, opd_runtime

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("teacher,student", [(False, False), (True, False), (False, True), (True, True)])
def test_deterministic_serving_is_independent_of_trainer_and_head(teacher, student):
    doc = specs.load(ROOT / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml").to_dict()
    doc["teacher"]["enable_deterministic_inference"] = teacher
    doc.setdefault("miles", {})["sglang_enable_deterministic_inference"] = student
    spec = specs.from_dict(doc)
    command = opd_runtime.teacher_command(spec, {"teacher": "/frozen-teacher"}, 12345)
    student_args = opd_runtime.with_native_overrides([], spec.document["miles"])
    assert ("--enable-deterministic-inference" in command) is teacher
    assert ("--sglang-enable-deterministic-inference" in student_args) is student
    assert "--enable-fp32-lm-head" not in command
    assert "--sglang-enable-fp32-lm-head" not in student_args
    assert "--deterministic-mode" not in student_args


def test_teacher_determinism_defaults_off_and_requires_boolean():
    doc = specs.load(ROOT / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml").to_dict()
    doc["teacher"].pop("enable_deterministic_inference")
    assert not specs.from_dict(doc).document["teacher"]["enable_deterministic_inference"]
    doc["teacher"]["enable_deterministic_inference"] = "false"
    with pytest.raises(ValueError, match="teacher.enable_deterministic_inference"):
        specs.from_dict(doc)


@pytest.mark.parametrize("minimum", ["0s", "1h"])
def test_opd_spec_preserves_allocated_or_unallocated_choice(minimum, monkeypatch):
    monkeypatch.delenv("MILES_CODE_OVERLAY", raising=False)
    doc = specs.load(ROOT / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml").to_dict()
    doc["launch"]["min_runtime"] = minimum
    spec = specs.from_dict(doc)
    rendered = opd_launch.specification("immutable-test-image", spec)
    assert len(rendered["tasks"]) == 1
    task = rendered["tasks"][0]
    assert task["resources"]["gpuCount"] == spec.allocation()["gpus_per_replica"]
    assert task["timeout"] == spec.launch["timeout"]
    if minimum == "0s":
        assert "minRuntime" not in task["context"]
    else:
        assert task["context"]["minRuntime"] == minimum
