"""Real verifier execution and adversarial checks for datasource qualification."""

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from scripts.miles import datasource_trials as trial


def tokenizer():
    return SimpleNamespace(
        chat_template="fixture template",
        apply_chat_template=lambda messages, **kwargs: "\n".join(message["content"] for message in messages)
        + " Assistant:",
        encode=lambda text, **kwargs: text.split(),
    )


def source_rows(task="math"):
    target = "42" if task == "math" else json.dumps({"func_name": "validate_lowercase", "N": None})
    return [
        dict(
            messages=[
                dict(role="user", content=f"Question {i}"),
                dict(role="assistant", content="SECRET REFERENCE ANSWER"),
            ],
            ground_truth=target,
        )
        for i in range(24)
    ]


def fixture_run(root, task="math"):
    spec = trial.task_spec(task)
    rows = trial.prepare_rows(source_rows(task), tokenizer(), spec)
    (root / "rollouts").mkdir()
    (root / "metrics").mkdir()
    hashes = {}
    for kind, selected in (("train", rows[:8]), ("eval", rows[8:])):
        raw = "".join(json.dumps(row) + "\n" for row in selected)
        (root / f"{kind}.jsonl").write_text(raw)
        hashes[kind] = hashlib.sha256(raw.encode()).hexdigest()
    (root / "preparation.json").write_text(
        json.dumps(dict(task=task, verifier=spec["verifier"], prepared_sha256=hashes))
    )
    (root / "arguments.json").write_text(
        json.dumps(["--rollout-max-response-len", "32", "--eval-max-response-len", "32"])
    )
    trial.write_registry(root, [spec["verifier"]])
    for name, version in (("0", 0), ("1", 1), ("eval_0", 0), ("eval_1", 2)):
        is_eval = name.startswith("eval")
        selected = rows[8:] if is_eval else rows[int(name) * 4 : (int(name) + 1) * 4]
        samples = []
        for row in selected:
            for index in range(1 if is_eval else 4):
                correct = index % 2 == 0
                response = (
                    (r"\boxed{42}" if correct else r"\boxed{43}")
                    if task == "math"
                    else ("hello" if correct else "HELLO")
                )
                samples.append(
                    dict(
                        prompt=row["input"],
                        label=row["label"],
                        metadata=copy.deepcopy(row["metadata"]),
                        response=response,
                        reward=float(correct),
                        tokens=[1, 2],
                        response_length=1,
                        rollout_log_probs=[-0.5],
                        weight_versions=[str(version)],
                    )
                )
        torch.save({"samples": samples}, root / f"rollouts/{name}.pt")
    (root / "metrics/publication.jsonl").write_text("".join(json.dumps({"version": i}) + "\n" for i in range(3)))


def test_real_verifiers_through_async_bridge(tmp_path):
    report = asyncio.run(trial.fixture_report(tmp_path))
    assert report["passed"] and len(report["cases"]) == 7
    assert {case["verifier"] for case in report["cases"]} == {"math", "ifeval", "ifeval_old"}
    assert any(case["expected"] == 0.5 for case in report["cases"])


@pytest.mark.parametrize("task", ["math", "ifeval"])
def test_preparation_strips_reference_and_preserves_targets(task):
    rows = source_rows(task)
    before = copy.deepcopy(rows)
    prepared = trial.prepare_rows(rows, tokenizer(), trial.task_spec(task))
    assert rows == before
    assert len(prepared) == 24 and all("SECRET" not in row["input"] for row in prepared)
    assert [row["metadata"]["source_row"] for row in prepared] == list(range(24))
    if task == "ifeval":
        assert json.loads(prepared[0]["label"])["func_name"] == "validate_lowercase"


@pytest.mark.parametrize("change", ["duplicate", "missing_user", "too_long", "wrong_math_target", "too_few"])
def test_preparation_rejects_ambiguous_inputs(change):
    rows = source_rows()
    if change == "duplicate":
        rows[-1] = rows[0]
    elif change == "missing_user":
        rows[0]["messages"] = [{"role": "assistant", "content": "solution"}]
    elif change == "too_long":
        rows[0]["messages"][0]["content"] = "long " * 2049
    elif change == "wrong_math_target":
        rows[0]["ground_truth"] = {"instruction_id": []}
    else:
        rows.pop()
    with pytest.raises(ValueError):
        trial.prepare_rows(rows, tokenizer(), trial.task_spec("math"))


def test_legacy_and_modern_if_schemas_are_not_interchangeable():
    with pytest.raises(ValueError, match="func_name"):
        trial.normalize_target({"instruction_id": ["keywords:existence"], "kwargs": []}, "ifeval_old")


@pytest.mark.parametrize("task", ["math", "ifeval"])
def test_independent_run_audit_with_real_verifiers(tmp_path, task):
    fixture_run(tmp_path, task)
    report = trial.audit(tmp_path)
    assert report["passed"] and report["results"]["0"]["mixed_reward_groups"] == 4
    assert report["results"]["eval_1"]["mean_reward"] == 1


@pytest.mark.parametrize(
    "change", ["wrong_reward", "stale", "nonfinite", "wrong_target", "wrong_prompts", "changed_input", "bad_length"]
)
def test_audit_rejects_corrupt_evidence(tmp_path, change):
    fixture_run(tmp_path)
    path = tmp_path / "rollouts/1.pt"
    data = torch.load(path, weights_only=False)
    first = data["samples"][0]
    if change == "wrong_reward":
        first["reward"] = 0.0
    elif change == "stale":
        first["weight_versions"] = ["0"]
    elif change == "nonfinite":
        first["rollout_log_probs"] = [float("nan")]
    elif change == "wrong_target":
        first["metadata"]["verifiers"][0]["target"] = "0"
    elif change == "wrong_prompts":
        data = torch.load(tmp_path / "rollouts/0.pt", weights_only=False)
        for sample in data["samples"]:
            sample["weight_versions"] = ["1"]
    elif change == "changed_input":
        with (tmp_path / "train.jsonl").open("a") as stream:
            stream.write("\n")
    else:
        first["response_length"] = 100
    torch.save(data, path)
    with pytest.raises(ValueError):
        trial.audit(tmp_path)


def test_local_snapshot_records_hash_without_claiming_remote_load(tmp_path, monkeypatch):
    raw = "".join(json.dumps(row) + "\n" for row in source_rows())
    path = tmp_path / "source.jsonl"
    path.write_text(raw)
    monkeypatch.setattr(trial.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer())
    report = trial.prepare(tmp_path / "prepared", "math", tmp_path / "hf", path)
    assert report["source"] == dict(
        kind="local_jsonl", path=str(path), sha256=hashlib.sha256(raw.encode()).hexdigest()
    )
    assert len((tmp_path / "prepared/eval.jsonl").read_text().splitlines()) == 16


def test_bounded_selection_preserves_actual_source_indices():
    rows = source_rows() + [dict(messages=[dict(role="user", content="Extra question")], ground_truth="42")]
    rows[2]["messages"][0]["content"] = "long " * 2049
    selected = trial.prepare_rows(rows, tokenizer(), trial.task_spec("math"), select_bounded=True)
    assert len(selected) == 24
    assert [row["metadata"]["source_row"] for row in selected] == [0, 1, *range(3, 25)]


def test_bounded_selection_does_not_scan_past_budget():
    bad = dict(messages=[dict(role="user", content="long " * 2049)], ground_truth="42")
    with pytest.raises(ValueError, match="24 unique"):
        trial.prepare_rows([bad] * 256 + source_rows(), tokenizer(), trial.task_spec("math"), select_bounded=True)


def test_local_profile_respects_model_context_and_disables_graphs(tmp_path):
    (tmp_path / "preparation.json").write_text(json.dumps(dict(task="math", hf="/tiny/hf", profile="local")))
    config = trial.configuration(tmp_path)
    config.validate()
    assert config.core.expert_parallel_size == 1
    assert config.core.max_sequence_length == config.miles["sglang_context_length"] == 512
    assert config.miles["rollout_max_response_len"] == config.miles["eval_max_response_len"] == 32
    assert config.miles["colocate"] and config.miles["sglang_disable_cuda_graph"]
    assert "sglang_cuda_graph_backend_decode" not in config.miles


@pytest.mark.parametrize(
    "constraint",
    [
        {"func_name": "arbitrary.import"},
        {"func_name": "validate_lowercase", "surprise": 1},
        {"func_name": "verify_keyword_frequency"},
    ],
)
def test_legacy_if_rejects_unknown_function_or_wrong_arguments(constraint):
    with pytest.raises(ValueError):
        trial.normalize_target(constraint, "ifeval_old")


@pytest.mark.parametrize("versions,valid", [([0, 1, 1, 2, 2], True), ([0, 1, 2], False), ([0, 1, 1, 2, 2, 2], False)])
def test_diagnostic_republication_is_not_an_optimizer_step(tmp_path, versions, valid):
    fixture_run(tmp_path, "ifeval")
    args_path = tmp_path / "arguments.json"
    argv = json.loads(args_path.read_text()) + ["--olmo-core-config", json.dumps({"diagnostic_interval": 1})]
    args_path.write_text(json.dumps(argv))
    (tmp_path / "metrics/publication.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "version": version,
                    "total_seconds": 0.1,
                    "repeated_version": index > 0 and versions[index - 1] == version,
                }
            )
            + "\n"
            for index, version in enumerate(versions)
        )
    )
    if valid:
        report = trial.audit(tmp_path)
        assert report["optimizer_steps"] == 2 and report["diagnostic_republications"] == 2
        assert report["diagnostic_republication_seconds"] == pytest.approx(0.2)
    else:
        with pytest.raises(ValueError, match="republications"):
            trial.audit(tmp_path)
