"""Historical probe summaries must use greedy tokens, not top-k tie ordering."""

import json

from scripts.miles import summarize_core_choices


def test_summary_recomputes_choices_from_retained_tokens(tmp_path):
    for name in ["provenance.json", "runtime.lock.json"]:
        (tmp_path / name).write_text("{}")
    rollout = {
        "row": 0,
        "domain": "math",
        "output_ids": [10],
        "argmax_ids": [20],
        "logprobs": [-0.7],
        "top2": [[[-0.7, 20], [-0.7, 10]]],
        "seconds": 0.5,
    }
    core = {"argmax_ids": [10], "logprobs": [-0.7], "top2_margin": [0.0], "chosen_rank": [1], "chosen_margin": [0.0]}
    detail = {"row": 0, "domain": "math", "full_sequence": {"core": core}, "prefix_at_a_time": {"core": core}}
    serving = {"engine_args": {}, "load_seconds": 1, "rollouts": [rollout], "forced": [rollout]}
    records = {"rollouts": [detail], "forced": [detail]}
    for arm in ["emo", "non-emo"]:
        directory = tmp_path / arm
        directory.mkdir()
        (directory / "core.json").write_text(
            json.dumps({"model": arm, "comparisons": {"auto": records, "full": records}})
        )
        for mode in ["auto", "full"]:
            (directory / f"{mode}.json").write_text(json.dumps(serving))
    report = summarize_core_choices.summarize(tmp_path, experiment="test-run", result_dataset="test-data")
    assert report["experiment"] == "test-run"
    assert report["workload"]["tokens_per_prompt"] == [1]
    for arm in report["arms"].values():
        for mode in arm["modes"].values():
            cached = mode["cached"]["prefix_at_a_time"]
            assert cached["argmax_matches"] == 1
            assert cached["greedy_token_vs_reported_top1_changes"] == 1
            assert cached["mean_abs_logprob"] == 0
            assert cached["strict_core_preference_disagreements"] == 0
            assert mode["timing"]["tokens_per_second"] == 2
