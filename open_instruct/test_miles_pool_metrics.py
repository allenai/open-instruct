"""Pool telemetry follows observations and completed training, independently of policy."""

import copy
import math

import pytest

from open_instruct import test_miles_pool_sampling, test_miles_prompt_replay
from open_instruct.miles.rollout import pool_metrics


def total(provider, metric):
    return sum(provider.metrics()[f"rollout/pool/{pool}/{metric}"] for pool in provider.providers)


@pytest.mark.parametrize("policy", test_miles_pool_sampling.POLICIES)
def test_selection_observation_commit_and_restore(policy):
    p = test_miles_pool_sampling.make(policy)
    samples = p.next_samples(8)
    assert total(p, "proposed") == 8
    assert total(p, "attempted") == total(p, "trained") == 0
    records = [
        test_miles_prompt_replay.record(samples[0], caps=2),
        test_miles_prompt_replay.record(samples[1], rewards=(1, 1)),
        test_miles_prompt_replay.record(samples[2], rewards=(-0.5, -0.5)),
        test_miles_prompt_replay.record(samples[3], valid=False),
        test_miles_prompt_replay.record(samples[4], rewards=(0, 1), filter_decision="passed"),
        test_miles_prompt_replay.record(samples[5], filter_decision="aborted"),
        test_miles_prompt_replay.record(samples[6], rewards=(0, 1), filter_decision="passed", tokens=None),
    ]
    p.feedback(records)
    restored = test_miles_pool_sampling.make(policy)
    restored.load_state_dict(p.state_dict())
    assert restored.metrics() == p.metrics()
    # Returned records may be delivered again after restoring pending selections.
    restored.feedback(records)
    assert restored.metrics() == p.metrics()
    p = restored
    assert total(p, "attempted") == total(p, "fresh_attempted") == 7
    assert total(p, "scored") == 5
    assert total(p, "invalid") == total(p, "aborted") == 1
    assert total(p, "filtered") == 4
    for suffix in ("all_zero", "all_one", "constant_nonzero", "other"):
        assert total(p, f"filtered_{suffix}") == 1
    assert total(p, "groups_with_truncation") == 1
    assert total(p, "truncated_responses") == 2
    assert total(p, "returned_responses") == 20
    assert total(p, "generated_tokens") == 1800
    assert total(p, "generated_tokens_unknown_groups") == 1
    assert total(p, "trained") == 0

    discarded = [{"kind": "disposition", "disposition": "discarded", **s.metadata} for s in (samples[5], samples[7])]
    p.feedback(discarded + discarded)
    assert total(p, "discarded") == 2
    assert total(p, "discarded_unobserved") == 1
    ids = [s.metadata["pool_selection_id"] for s in (samples[4], samples[6])]
    p.commit(0, ids)
    p.commit(0, ids)
    assert total(p, "trained") == total(p, "fresh_groups") == 2
    assert total(p, "trained_tokens") == 200
    assert total(p, "trained_tokens_unknown_groups") == 1
    assert total(p, "pending") == 0
    for field in ("proposed_share", "attempted_share", "trained_share", "generated_token_share"):
        assert total(p, field) == pytest.approx(1)
    metrics = p.metrics()
    for pool in ("math", "if", "empty"):
        prefix = f"rollout/pool/{pool}/"
        rows = metrics[prefix + "available_rows"]
        assert metrics[prefix + "coverage"] == metrics[prefix + "fresh_attempted"] / max(1, rows)
    assert all(math.isfinite(v) for v in metrics.values())
    assert all(v == 0 for k, v in metrics.items() if k.startswith("rollout/pool/empty/"))
    restored = test_miles_pool_sampling.make(policy)
    restored.load_state_dict(p.state_dict())
    assert restored.metrics() == metrics
    broken = copy.deepcopy(p.state_dict())
    broken["observations"]["math/attempted"] = -1
    with pytest.raises(ValueError, match="observations"):
        restored.load_state_dict(broken)
    assert restored.metrics() == metrics


def test_replay_counts_do_not_inflate_coverage_and_selection_rollback():
    replay = dict(fraction=1, window=1, max_successes=0, truncation="any", length_drop=0.1)
    p = test_miles_pool_sampling.make("shuffle", (8, 0), replay)
    first = p.next_samples(1)[0]
    p.feedback([test_miles_prompt_replay.record(first, caps=4)])
    shorter = p.next_samples(1)[0]
    p.feedback([test_miles_prompt_replay.record(shorter, rewards=(1, 1, 1, 1), tokens=50)])
    before = p.metrics()
    state = p.selection_state_dict()
    sample = p.next_samples(1)[0]
    assert sample.metadata["prompt_replay_review"]
    assert total(p, "replay_proposed") == 1
    p.load_selection_state_dict(state)
    assert p.metrics() == before
    assert p.next_samples(1)[0] == sample
    p.feedback([test_miles_prompt_replay.record(sample, rewards=(0, 1), filter_decision="passed")])
    p.commit(0, [sample.metadata["pool_selection_id"]])
    assert total(p, "fresh_proposed") == total(p, "fresh_attempted") == 2
    assert total(p, "replay_proposed") == total(p, "replay_attempted") == total(p, "replay_groups") == 1
    assert p.metrics()["rollout/pool/math/coverage"] == before["rollout/pool/math/coverage"] == 0.25


def test_dashboard_groups_metrics_by_measurement_and_uses_completed_update():
    assert pool_metrics.dashboard(
        {
            "rollout/pool/math/trained": 3,
            "rollout/pool/if/trained": 7,
            "rollout/pool/math/replay/candidates": 2,
            "other": 99,
        },
        12,
    ) == {"pool/update": 12, "pool/trained/math": 3, "pool/trained/if": 7}
