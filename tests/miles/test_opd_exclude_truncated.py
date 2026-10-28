"""OPD's distillation.exclude_truncated switch: validation, run identity and the hook's loss mask."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from miles.utils.types import Sample

from open_instruct.miles.configuration import specs
from open_instruct.miles.distillation import opd_hooks
from open_instruct.miles.execution import workflow

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/miles/fixtures/opd/qwen35-4b-tiny.toml"


def load(**distillation):
    document = specs.load(FIXTURE).to_dict()
    document["distillation"].pop("exclude_truncated", None)
    document["distillation"].update(distillation)
    return document


def test_exclude_truncated_defaults_off_without_changing_existing_run_identity():
    document = specs.from_dict(load()).to_dict()
    assert document["distillation"]["exclude_truncated"] is False
    recorded = {
        **document,
        "distillation": {k: v for k, v in document["distillation"].items() if k != "exclude_truncated"},
    }
    assert workflow.fingerprint(document) == workflow.fingerprint(recorded)


def test_exclude_truncated_is_part_of_run_identity():
    kept = specs.from_dict(load()).to_dict()
    excluded = specs.from_dict(load(exclude_truncated=True)).to_dict()
    assert excluded["distillation"]["exclude_truncated"] is True
    assert workflow.fingerprint(excluded) != workflow.fingerprint(kept)


def test_exclude_truncated_must_be_boolean():
    with pytest.raises(ValueError, match="distillation.exclude_truncated"):
        specs.from_dict(load(exclude_truncated="yes"))


@pytest.mark.parametrize("exclude", [False, True])
def test_post_process_removes_only_truncated_samples_when_enabled(monkeypatch, tmp_path, exclude):
    async def scores(*args, **kwargs):
        return {"meta_info": {"input_token_logprobs": [[None, 1], [-1.0, 2], [-2.0, 3]]}}

    monkeypatch.delenv("OI_OPD_EOPD_TOP_K", raising=False)
    monkeypatch.setenv("OI_OPD_OUTPUT", str(tmp_path))
    monkeypatch.setenv(opd_hooks.EXCLUDE_TRUNCATED_ENV, "1" if exclude else "0")
    monkeypatch.setattr(opd_hooks.on_policy_distillation, "reward_func", scores)
    monkeypatch.setattr(opd_hooks, "_LIMIT", None)
    samples = []
    for index, status in enumerate((Sample.Status.COMPLETED, Sample.Status.TRUNCATED)):
        sample = Sample(index=index, tokens=[1, 2, 3], response_length=2, response="ab", status=status)
        sample.reward = asyncio.run(opd_hooks.reward(SimpleNamespace(), sample))
        samples.append(sample)
    assert opd_hooks.post_process(SimpleNamespace(reward_key=None), samples) == ([0.0, 0.0], [0.0, 0.0])
    # The teacher still scores the truncated response; only its loss mask is dropped.
    assert all(sample.teacher_log_probs is not None for sample in samples)
    assert [sample.remove_sample for sample in samples] == [False, exclude]
