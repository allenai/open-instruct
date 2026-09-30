"""Selection and staleness contrasts retain pairing and question-level uncertainty."""

import hashlib
import json

import pytest
from scripts.miles import compare_opd_controls


def fixture():
    arms = {name: {} for name in compare_opd_controls.ARMS}
    # Only the combined intervention succeeds: pure interaction, no effect at
    # the fresh/admitted baseline. Two repetitions are still one question.
    for repeat in range(2):
        key = ("dapo", "q0", "sampled", repeat)
        for arm in arms:
            arms[arm][key] = dict(
                prompt="Q", label="7", input_ids=[1], seed=repeat, reward=int(arm == "selected-p4"), status="completed"
            )
    return arms


def test_factorial_interaction_is_not_mistaken_for_either_baseline_effect():
    result = compare_opd_controls.summarize(fixture())["dapo/sampled"]
    assert result["questions"] == 1 and result["responses_per_arm"] == 2
    contrasts = result["contrasts"]
    assert contrasts["selection_at_age_zero"]["accuracy_delta"] == 0
    assert contrasts["staleness_on_admitted_prompts"]["accuracy_delta"] == 0
    assert contrasts["selection_by_staleness_interaction"]["accuracy_delta"] == 1
    assert contrasts["selection_mean_over_age_settings"]["accuracy_delta"] == 0.5


def test_missing_answers_or_changed_sampling_seed_cannot_form_a_contrast():
    arms = fixture()
    arms["selected-p4"].pop(next(iter(arms["selected-p4"])))
    with pytest.raises(ValueError, match="matched"):
        compare_opd_controls.summarize(arms)
    arms = fixture()
    next(iter(arms["selected-p4"].values()))["seed"] = 99
    with pytest.raises(ValueError, match="seeds"):
        compare_opd_controls.summarize(arms)


def test_artifact_comparison_checks_server_settings_and_completed_hashes(tmp_path):
    paths = {}
    for name, rows in fixture().items():
        root = tmp_path / name
        root.mkdir()
        paths[name] = root
        records = [
            dict(row, dataset=key[0], question_id=key[1], mode=key[2], repeat=key[3]) for key, row in rows.items()
        ]
        raw = "".join(json.dumps(row) + "\n" for row in records).encode()
        (root / "responses.jsonl").write_bytes(raw)
        (root / "complete.json").write_text(
            json.dumps(dict(responses=2, responses_sha256=hashlib.sha256(raw).hexdigest()))
        )
        provenance = dict(
            panel_sha256="panel",
            repeats=2,
            head="fp32",
            sampling_temperature=1,
            response_cap=16384,
            request_order_sha256="seeds",
            expected_responses=2,
            command=["python", "-m", "sglang.launch_server", "--model-path", name, "--enable-deterministic-inference"],
        )
        (root / "provenance.json").write_text(json.dumps(provenance))
    assert compare_opd_controls.compare(paths)["datasets"]["dapo/sampled"]["pass_at_1"]["selected-p4"] == 1
    path = paths["selected-p4"] / "provenance.json"
    changed = json.loads(path.read_text())
    changed["command"].remove("--enable-deterministic-inference")
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="settings differ"):
        compare_opd_controls.compare(paths)
    changed["command"].append("--enable-deterministic-inference")
    path.write_text(json.dumps(changed))
    (paths["selected-p4"] / "responses.jsonl").write_text("tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        compare_opd_controls.compare(paths)
