"""Domain-balanced token mean: per-sample weights, the hook that attaches them, and the run-file mapping."""

from types import SimpleNamespace

import pytest

from open_instruct.miles.configuration import specs
from open_instruct.miles.errors import InputError
from open_instruct.miles.megatron_grpo import domain_balance, grpo_args, grpo_hooks
from open_instruct.test_miles_megatron_grpo import PREPARED, qwen35_async_dppo


def sample(domain, length, *, removed=False, loss_mask=None, reward=0.0, index=0):
    return SimpleNamespace(
        index=index,
        group_index=index // 2,
        reward=reward,
        tokens=[1, 2, *range(length)],
        response_length=length,
        response="answer",
        status=SimpleNamespace(value="completed"),
        metadata={"domain": domain, "verifiers": []},
        remove_sample=removed,
        loss_mask=loss_mask,
        train_metadata=None,
    )


def test_weights_give_each_domain_an_equal_share_of_a_token_mean():
    samples = [sample("math", 8000), sample("math", 4000), sample("ifeval", 300), sample("ifeval", 100)]
    weights = domain_balance.weights(samples)
    weighted = {}
    for item, weight in zip(samples, weights, strict=True):
        weighted[item.metadata["domain"]] = weighted.get(item.metadata["domain"], 0.0) + weight * item.response_length
    total = sum(item.response_length for item in samples)
    assert weighted == pytest.approx({"math": total / 2, "ifeval": total / 2})
    assert sum(w * s.response_length for w, s in zip(weights, samples, strict=True)) == pytest.approx(total)


def test_weights_count_only_training_tokens_and_skip_empty_domains():
    samples = [
        sample("math", 100),
        sample("math", 100, removed=True),
        sample("ifeval", 50, loss_mask=[1] * 20 + [0] * 30),
        sample("code", 40, removed=True),
    ]
    weights = domain_balance.weights(samples)
    # 120 training tokens over two active domains: math 100 -> 0.6, ifeval 20 -> 3.0; code has none.
    assert weights == pytest.approx([0.6, 0.6, 3.0, 1.0])
    assert domain_balance.weights([sample("math", 10), sample("math", 30)]) == [1.0, 1.0]


def test_weights_require_a_domain_on_every_sample():
    unlabeled = sample("math", 10)
    unlabeled.metadata = {"verifiers": []}
    with pytest.raises(ValueError, match="domain"):
        domain_balance.weights([sample("math", 10), unlabeled])


def test_attach_keeps_existing_train_metadata_and_batch_weights_reads_it_back():
    samples = [sample("math", 30), sample("ifeval", 10)]
    samples[0].train_metadata = {"policy_refresh": {"version": 3}}
    assert domain_balance.attach(samples) == pytest.approx({"math": 40 / 60, "ifeval": 40 / 20})
    assert samples[0].train_metadata["policy_refresh"] == {"version": 3}
    batch = {"metadata": [item.train_metadata for item in samples]}
    assert domain_balance.batch_weights(batch) == pytest.approx([40 / 60, 2.0])
    with pytest.raises(ValueError, match=domain_balance.WEIGHT_KEY):
        domain_balance.batch_weights({"metadata": [{}, samples[1].train_metadata]})
    with pytest.raises(ValueError, match=domain_balance.WEIGHT_KEY):
        domain_balance.batch_weights({})


def test_grpo_post_process_attaches_weights_only_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv(grpo_hooks.OUTPUT_ENV, str(tmp_path))
    args = SimpleNamespace(n_samples_per_prompt=2, grpo_std_normalization=False)

    def batch():
        return [
            sample("math", 300, reward=1.0, index=0),
            sample("math", 300, index=1),
            sample("ifeval", 100, reward=1.0, index=2),
            sample("ifeval", 100, index=3),
        ]

    samples = batch()
    grpo_hooks.post_process(args, samples)
    assert all(item.train_metadata is None for item in samples)
    monkeypatch.setenv(domain_balance.ENV, "1")
    samples = batch()
    raw, centered = grpo_hooks.post_process(args, samples)
    assert centered == [0.5, -0.5, 0.5, -0.5]
    assert [item.train_metadata[domain_balance.WEIGHT_KEY] for item in samples] == pytest.approx(
        [800 / 1200, 800 / 1200, 2.0, 2.0]
    )


def test_domain_aggregation_keeps_the_token_mean_flag_and_enables_the_weights(tmp_path):
    doc = qwen35_async_dppo()
    doc["training"]["loss_aggregation"] = "domain"
    doc["output"]["root"] = str(tmp_path / "root")
    spec = specs.from_dict(doc)
    assert "--calculate-per-token-loss" in grpo_args.native_arguments(spec, PREPARED, "/converted/native", [])
    assert domain_balance.environment(spec.document["training"]["loss_aggregation"]) == {domain_balance.ENV: "1"}
    assert domain_balance.environment("token") == {}


def test_domain_aggregation_requires_the_dppo_loss():
    doc = qwen35_async_dppo()
    doc["training"]["loss_aggregation"] = "domain"
    doc["objective"]["policy_loss"] = "ppo"
    with pytest.raises(InputError, match="requires objective.policy_loss"):
        specs.from_dict(doc)


def test_runtime_preflight_tests_do_not_inherit_the_run_switch():
    environment = {domain_balance.ENV: "1", "OI_DPPO_THRESHOLD": "0.1"}
    assert domain_balance.test_environment(environment) == {"OI_DPPO_THRESHOLD": "0.1"}
    assert environment[domain_balance.ENV] == "1"
