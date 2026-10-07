"""The paired precision option reaches both model construction and serving argv."""

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.configuration.run_spec import RunSpec
from open_instruct.miles.errors import InputError


def test_fp32_head_enables_matching_serving_option():
    config = RunConfig(CoreConfig(fp32_lm_head=True), {})
    assert config.resolved_miles()["sglang_enable_fp32_lm_head"] is True
    assert "sglang_enable_fp32_lm_head" not in RunConfig(CoreConfig(), {}).resolved_miles()


def test_fp32_head_rejects_explicit_serving_conflict_and_bad_type():
    with pytest.raises(InputError, match="fp32_lm_head conflicts"):
        RunConfig(CoreConfig(fp32_lm_head=True), {"sglang_enable_fp32_lm_head": False}).resolved_miles()
    with pytest.raises(InputError, match="core.fp32_lm_head"):
        CoreConfig(fp32_lm_head="true")


def test_structured_fp32_head_option():
    spec = RunSpec.load("configs/miles/examples/small.toml", ["trainer.fp32_lm_head=true"])
    assert spec.compile().core.fp32_lm_head
    assert "--sglang-enable-fp32-lm-head" in spec.compile().arguments()
    assert spec.compile().resolved_miles()["sglang_enable_fp32_lm_head"] is True
