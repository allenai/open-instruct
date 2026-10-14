"""Keep update horizons and counting units explicit in the offline comparison."""

import pytest
from scripts.miles import analyze_opd_selection


def log_row(index, length, count):
    metrics = {
        "rollout/episode_response_length/mean": length,
        "rollout/num_training_samples": count,
        "rollout/truncated_ratio": 0.5,
    }
    return f"prefix - perf {index}: {metrics!r}\n"


def test_comparison_matches_ids_and_weights_by_responses(tmp_path):
    sync, asynchronous = tmp_path / "sync.log", tmp_path / "async.log"
    sync.write_text(log_row(0, 1000, 4) + log_row(1, 100, 2) + log_row(2, 200, 6))
    asynchronous.write_text(log_row(1, 80, 2) + log_row(2, 160, 6))
    result = analyze_opd_selection.compare(sync, asynchronous)
    assert result["unmatched_updates"] == {"sync": [0], "async": []}
    window = result["windows"]["all_matched"]
    assert window["rollout_ids"] == [1, 2]
    assert window["sync"]["response_length_mean"] == 175
    assert window["async"]["response_length_mean"] == 140
    assert window["sync"]["aborted_groups_filtered"] is None


def test_duplicate_update_requires_explicit_attempt_selection(tmp_path):
    path = tmp_path / "duplicate.log"
    path.write_text(log_row(0, 1, 2) * 2)
    with pytest.raises(ValueError, match="Duplicate rollout"):
        analyze_opd_selection.read_batches(path)
