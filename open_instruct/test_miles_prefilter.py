"""Pre-filter observation reports every judged group by domain and dumps sampled groups."""

import json
import random
from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration.config import CoreConfig, RunConfig
from open_instruct.miles.rollout import prefilter


class Output:
    def __init__(self, keep, reason=None):
        self.keep, self.reason = keep, reason


def drop_constant(args, samples, **kwargs):
    rewards = {sample.reward for sample in samples}
    return Output(len(rewards) > 1, None if len(rewards) > 1 else "zero_std")


INNER = f"{__name__}.drop_constant"


def group(verifier, rewards, *, truncated=(), ungraded=(), length=10, targets=None, group_index=0):
    samples = []
    for index, reward in enumerate(rewards):
        metadata = {
            "verifiers": [{"name": verifier, "target": targets if targets is not None else {"answer": 1}}],
            "prepared_sample_id": f"p{group_index}",
            "source_dataset": "fixture",
            "verifier_diagnostics": {verifier: {"status": "ok", "outcomes": "pf"}},
        }
        if index in ungraded:
            metadata["reward_omission"] = {"verifier": verifier, "error": "boom"}
        samples.append(
            SimpleNamespace(
                group_index=group_index,
                index=index,
                rollout_id=None,
                prompt="Write code",
                response=f"response {index}",
                response_length=length,
                reward=reward,
                status=SimpleNamespace(value="truncated" if index in truncated else "completed"),
                weight_versions=[[{"version": "3", "abs_start": 0, "abs_end": length}]],
                metadata=metadata,
            )
        )
    return samples


def run_args(tmp_path=None, **core):
    return SimpleNamespace(
        prefilter_inner_filter_path=INNER,
        reward_key=None,
        save=str(tmp_path) if tmp_path is not None else None,
        olmo_core=CoreConfig(**core),
    )


@pytest.fixture(autouse=True)
def fresh_observer(monkeypatch):
    monkeypatch.setattr(prefilter, "OBSERVER", prefilter.Observer(rng=random.Random(0)))


def test_metrics_count_every_judged_group_before_the_filter_drops_any():
    args = run_args()
    groups = [
        group("code", [0, 0, 0, 0]),
        group("code", [1, 1, 1, 1]),
        group("code", [0, 1, 0, 1], truncated={0}),
        group("code", [0, 1, 1, 1], ungraded={0}),  # graded rewards are all 1
        group("math", [0.5, 0.5]),
    ]
    outputs = [prefilter.observed_filter(args, samples) for samples in groups]
    # The configured filter's decisions pass through unchanged.
    assert [output.keep for output in outputs] == [False, False, True, True, False]
    metrics = prefilter.drain(7)
    assert metrics["rollout/prefilter/code/groups"] == 4
    assert metrics["rollout/prefilter/code/all_zero_fraction"] == 0.25
    assert metrics["rollout/prefilter/code/all_one_fraction"] == 0.5
    assert metrics["rollout/prefilter/code/mixed_fraction"] == 0.25
    assert metrics["rollout/prefilter/code/all_equal_fraction"] == 0.75
    assert metrics["rollout/prefilter/code/kept_fraction"] == 0.5
    assert metrics["rollout/prefilter/code/truncated_fraction"] == 1 / 16
    # 15 graded rewards; the ungraded placeholder zero is left out.
    assert metrics["rollout/prefilter/code/reward_mean"] == pytest.approx(9 / 15)
    assert metrics["rollout/prefilter/math/constant_fraction"] == 1
    assert metrics["rollout/prefilter/all/groups"] == 5
    assert metrics["rollout/prefilter/all/kept_fraction"] == 0.4
    assert "rollout/prefilter/dumped_groups" not in metrics
    # Each update reports only the groups judged since the previous drain.
    assert prefilter.drain(8) == {}


def test_without_a_configured_filter_every_group_is_kept_and_observed():
    args = run_args()
    args.prefilter_inner_filter_path = None
    assert prefilter.observed_filter(args, group("code", [1, 1])) is True
    assert prefilter.drain(1)["rollout/prefilter/code/kept_fraction"] == 1


def test_observation_errors_never_change_the_filter_decision(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("observer bug")

    monkeypatch.setattr(prefilter.OBSERVER, "observe", broken)
    output = prefilter.observed_filter(run_args(), group("code", [0, 1]))
    assert output.keep
    assert prefilter.drain(1)["rollout/prefilter/observation_failures"] == 1


def test_dump_selects_domains_caps_each_update_and_appends_across_restarts(tmp_path, monkeypatch):
    args = run_args(tmp_path, sample_dump_fraction=1.0, sample_dump_domains=["code"], sample_dump_max_groups=2)
    for index in range(3):
        prefilter.observed_filter(args, group("code", [0, 0], group_index=index))
    prefilter.observed_filter(args, group("math", [0, 1], group_index=9))
    assert prefilter.drain(4)["rollout/prefilter/dumped_groups"] == 2

    # A resumed process appends its own attempt to the same file.
    monkeypatch.setattr(prefilter, "OBSERVER", prefilter.Observer(rng=random.Random(0)))
    prefilter.observed_filter(args, group("code", [1, 0], truncated={1}, group_index=5))
    prefilter.drain(5)

    rows = [json.loads(line) for line in (tmp_path / prefilter.DUMP_FILE).read_text().splitlines()]
    assert [(row["logged_rollout_id"], row["prompt_id"]) for row in rows] == [(4, "p0"), (4, "p1"), (5, "p5")]
    assert rows[0]["attempt"] != rows[2]["attempt"]
    assert rows[0]["domain"] == "code" and rows[0]["group_outcome"] == "all_zero"
    assert rows[0]["filter_kept"] is False and rows[0]["filter_reason"] == "zero_std"
    assert rows[0]["targets"] == {"code": {"answer": 1}}
    last = rows[2]["responses"]
    assert [response["response"] for response in last] == ["response 0", "response 1"]
    assert [response["finish"] for response in last] == ["completed", "truncated"]
    assert last[0]["reward"] == 1.0 and last[0]["response_tokens"] == 10
    assert last[0]["policy_versions"] == ["3"]
    assert last[0]["verifier_diagnostics"] == {"code": {"status": "ok", "outcomes": "pf"}}


def test_dump_fraction_samples_groups_and_large_targets_are_omitted(tmp_path):
    args = run_args(tmp_path, sample_dump_fraction=0.25, sample_dump_max_groups=1000)
    big = {"tests": ["x" * 100] * 1000}
    for index in range(400):
        prefilter.observed_filter(args, group("code", [0, 1], targets=big, group_index=index))
    dumped = prefilter.drain(1)["rollout/prefilter/dumped_groups"]
    assert 60 < dumped < 140
    row = json.loads((tmp_path / prefilter.DUMP_FILE).read_text().splitlines()[0])
    assert "targets" not in row and row["targets_omitted_chars"] > prefilter.MAX_TARGET_CHARS


def test_dump_is_off_by_default(tmp_path):
    prefilter.observed_filter(run_args(tmp_path), group("code", [0, 1]))
    prefilter.drain(1)
    assert not (tmp_path / prefilter.DUMP_FILE).exists()


@pytest.mark.parametrize(
    ("core", "match"),
    [
        ({"sample_dump_fraction": 1.5}, "sample_dump_fraction"),
        ({"sample_dump_fraction": -0.1}, "sample_dump_fraction"),
        ({"sample_dump_max_groups": 0}, "sample_dump_max_groups"),
        ({"sample_dump_domains": "code"}, "sample_dump_domains"),
    ],
)
def test_invalid_dump_settings_are_rejected(core, match):
    with pytest.raises(Exception, match=match):
        CoreConfig(**core)


def test_dump_requires_a_save_directory():
    options = dict(hf_checkpoint="/fixture/hf", global_batch_size=4, rollout_batch_size=2, n_samples_per_prompt=2)
    with pytest.raises(Exception, match="requires miles.save"):
        RunConfig(CoreConfig(sample_dump_fraction=0.01), options).validate()
    RunConfig(CoreConfig(sample_dump_fraction=0.01), options | {"save": "/fixture/save"}).validate()
