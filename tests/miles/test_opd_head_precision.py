"""Keep independently configured teacher and student serving precision explicit."""

from pathlib import Path

import pytest

from open_instruct.miles.configuration import specs
from open_instruct.miles.distillation import opd_runtime

ROOT = Path(__file__).resolve().parents[2]


def document():
    return specs.load(ROOT / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml").to_dict()


@pytest.mark.parametrize("teacher_fp32,student_fp32", [(False, False), (True, False), (False, True), (True, True)])
def test_teacher_and_student_precision_are_independent(teacher_fp32, student_fp32):
    doc = document()
    doc["teacher"]["enable_fp32_lm_head"] = teacher_fp32
    doc.setdefault("miles", {})["sglang_enable_fp32_lm_head"] = student_fp32
    spec = specs.from_dict(doc)
    command = opd_runtime.teacher_command(spec, {"teacher": "/frozen-teacher"}, 12345)
    assert ("--enable-fp32-lm-head" in command) is teacher_fp32
    assert command[command.index("--model-path") + 1] == "/frozen-teacher"
    student_args = opd_runtime.with_native_overrides([], spec.document["miles"])
    assert ("--sglang-enable-fp32-lm-head" in student_args) is student_fp32


def test_teacher_precision_defaults_off_and_rejects_strings():
    doc = document()
    doc["teacher"].pop("enable_fp32_lm_head", None)
    assert not specs.from_dict(doc).document["teacher"]["enable_fp32_lm_head"]
    doc["teacher"]["enable_fp32_lm_head"] = "false"
    with pytest.raises(ValueError, match="teacher.enable_fp32_lm_head"):
        specs.from_dict(doc)
