"""CPU preparation contracts: no dataset download, GPU, Ray or model load."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from open_instruct.miles.datasets import run_data
from open_instruct.miles.errors import InputError
from open_instruct.miles.rewards import task_verifiers


class Tokenizer:
    chat_template = "HF template"

    def __init__(self):
        self.rendered = []

    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.rendered.append(copy.deepcopy(messages))
        return chat_template + "|" + "|".join(message["content"] for message in messages) + "|assistant:"

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


@pytest.fixture
def environment(tmp_path, monkeypatch):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "tokenizer_config.json").write_text('{"chat_template":"HF template"}')
    tokenizer = Tokenizer()
    monkeypatch.setattr(run_data, "_tokenizer", lambda path: tokenizer)
    return hf, tmp_path / "prepared", tokenizer


def prepare(environment, data):
    hf, output, _ = environment
    return run_data.prepare_data(data, hf, output, max_prompt_length=2000, seed=17)


def source_manifest(tmp_path, *, template=True):
    root = tmp_path / "source"
    root.mkdir()
    artifacts = {}
    for split in ("train", "eval"):
        row = {
            "messages": [{"role": "user", "content": f"{split} original wrapped prompt"}],
            "ground_truth": "42",
            "metadata": {
                "prepared_sample_id": split,
                "verifiers": [{"name": "gsm8k", "target": "42", "weight": 2.5}],
                "original": {"keep": True},
            },
        }
        raw = json.dumps(row).encode() + b"\n"
        (root / f"{split}.jsonl").write_bytes(raw)
        artifacts[split] = {"path": f"{split}.jsonl", "sha256": hashlib.sha256(raw).hexdigest(), "records": 1}
    options = {
        "input_key": "messages",
        "label_key": "ground_truth",
        "metadata_key": "metadata",
        "apply_chat_template": True,
        "custom_rm_path": "olmo_miles.rl.rewards.registered_reward",
        "chat_template": None,
    }
    if template:
        raw = b"original template"
        (root / "chat.jinja").write_bytes(raw)
        options["chat_template"] = {
            "name": "baseline",
            "path": "chat.jinja",
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
    manifest = {"schema_version": 1, "miles": options, "artifacts": artifacts}
    path = root / "rl-manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_manifest_adoption_preserves_template_targets_and_resume(environment, tmp_path):
    path = source_manifest(tmp_path)
    data = {"rl_manifest": str(path), "shuffle": True, "seed": 29}
    result = prepare(environment, data)
    row = json.loads(Path(result["prompt_data"]).read_text())
    assert row["input"] == "original template|train original wrapped prompt|assistant:"
    assert row["metadata"]["original"] == {"keep": True}
    assert row["metadata"]["verifiers"] == [{"name": "gsm8k", "target": "42", "weight": 2.5}]
    before = len(environment[2].rendered)
    assert prepare(environment, data) == result
    assert len(environment[2].rendered) == before
    registry = json.loads(Path(result["reward_config"]).read_text())
    assert registry["gsm8k"]["factory"] == "open_instruct.miles.rewards.verifiers.GSM8KVerifier"


@pytest.mark.parametrize("target", ["input", "output", "tokenizer"])
def test_resume_rejects_changed_inputs_or_outputs(environment, tmp_path, target):
    path = source_manifest(tmp_path)
    data = {"rl_manifest": str(path)}
    result = prepare(environment, data)
    changed = {
        "input": path.parent / "train.jsonl",
        "output": Path(result["prompt_data"]),
        "tokenizer": environment[0] / "tokenizer_config.json",
    }[target]
    changed.write_text(changed.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        prepare(environment, data)


def test_manifest_rejects_digest_before_completion(environment, tmp_path):
    path = source_manifest(tmp_path)
    (path.parent / "eval.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare(environment, {"rl_manifest": str(path)})
    assert not environment[1].exists()


def test_named_tasks_strip_reference_answers_and_select_disjoint(environment, monkeypatch):
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": "SECRET SOLUTION"},
            ],
            "ground_truth": [str(i)],
        }
        for i in range(12)
    ]
    monkeypatch.setattr(run_data, "_source_rows", lambda name: rows)
    result = prepare(
        environment, {"tasks": [{"task": "gsm8k", "train_count": 5, "eval_count": 3, "prompt_wrapper": "auto"}]}
    )
    train = [json.loads(line) for line in Path(result["prompt_data"]).read_text().splitlines()]
    evaluation = [json.loads(line) for line in Path(result["eval_prompt_data"][1]).read_text().splitlines()]
    assert len(train) == 5 and len(evaluation) == 3
    assert not {row["metadata"]["source_row"] for row in train} & {row["metadata"]["source_row"] for row in evaluation}
    assert all("SECRET" not in row["input"] and run_data.ANSWER_PREFIX in row["input"] for row in train)
    assert len(environment[2].rendered) == 8


@pytest.mark.parametrize(
    "task,dataset",
    [
        ("gsm8k", "ai2-adapt-dev/rlvr_gsm8k_zs"),
        ("gsm8k_original", "ai2-adapt-dev/rlvr_gsm8k_zs"),
        ("gsm8k-less-noise", "techarb/gsm8k-cleaner"),
    ],
)
def test_gsm8k_variants_share_verifier_and_record_distinct_sources(environment, monkeypatch, task, dataset):
    assert run_data.TASKS[task][0] == dataset
    assert run_data.TASKS[task][2] == "gsm8k"
    assert run_data.TASKS["gsm8k_original"] == run_data.TASKS["gsm8k"]
    rows = [
        {"messages": [{"role": "user", "content": f"Question {i}"}], "ground_truth": [str(i)], "original_row": 10 + i}
        for i in range(4)
    ]
    loaded = []
    monkeypatch.setattr(run_data, "_source_rows", lambda name: loaded.append(name) or rows)
    result = prepare(environment, {"tasks": [{"task": task, "train_count": 4}]})
    train = [json.loads(line) for line in Path(result["prompt_data"]).read_text().splitlines()]
    assert loaded == [task]
    assert {row["metadata"]["original_row"] - row["metadata"]["source_row"] for row in train} == {10}
    assert {row["metadata"]["source_dataset"] for row in train} == {dataset}
    assert all(row["metadata"]["prepared_sample_id"].startswith(f"{task}:train:") for row in train)
    registry = json.loads(Path(result["reward_config"]).read_text())
    assert registry == {"gsm8k": {"factory": run_data.FACTORIES["gsm8k"]}}


def test_less_noise_preserves_cleaned_seed_convention(environment, monkeypatch):
    rows = [{"question": f"Question {i}", "ground_truth": str(i), "original_row": 100 + i} for i in range(12)]
    monkeypatch.setattr(run_data, "_source_rows", lambda name: rows)
    result = prepare(environment, {"tasks": [{"task": "gsm8k-less-noise", "train_count": 5, "eval_count": 3}]})
    # The historical gsm8k seed convention remains; held-out rows now come from
    # a separate draw over the remainder, followed by the usual split shuffling.
    for path, indices in [(result["prompt_data"], [6, 5, 0, 10, 7]), (result["eval_prompt_data"][1], [2, 11, 8])]:
        prepared = [json.loads(line) for line in Path(path).read_text().splitlines()]
        assert [row["metadata"]["source_row"] for row in prepared] == indices
        assert [row["metadata"]["original_row"] for row in prepared] == [100 + i for i in indices]
        assert [row["label"] for row in prepared] == [str(i) for i in indices]


def test_prepared_cleaned_gsm8k_resumes_without_changing_dataset(environment, monkeypatch):
    monkeypatch.setattr(run_data, "_source_rows", lambda name: [{"question": "Question", "ground_truth": "42"}])
    data = {"tasks": [{"task": "gsm8k", "train_count": 1}]}
    with monkeypatch.context() as historical:
        historical.setitem(run_data.TASKS, "gsm8k", run_data.TASKS["gsm8k-less-noise"])
        result = prepare(environment, data)
    before = Path(result["prompt_data"]).read_bytes()

    def unexpected_download(name):
        raise AssertionError("Resume must reuse the immutable prepared dataset")

    monkeypatch.setattr(run_data, "_source_rows", unexpected_download)
    assert prepare(environment, data) == result
    assert Path(result["prompt_data"]).read_bytes() == before
    assert json.loads(before)["metadata"]["source_dataset"] == "techarb/gsm8k-cleaner"


def test_generated_multiplication_is_reproducible_and_preserves_reward_weights(environment, tmp_path):
    data = {"tasks": [{"task": "multiplication", "train_count": 8, "eval_count": 4}]}
    first = prepare(environment, data)
    second_env = (environment[0], tmp_path / "another", environment[2])
    second = prepare(second_env, data)
    assert Path(first["prompt_data"]).read_bytes() == Path(second["prompt_data"]).read_bytes()
    row = json.loads(Path(first["prompt_data"]).read_text().splitlines()[0])
    assert [spec["weight"] for spec in row["metadata"]["verifiers"]] == [10.0, 1.0]
    answer = row["label"]
    assert (
        asyncio.run(run_data.MultiplicationVerifier().async_call([], f"<answer>{answer}</answer>", answer)).score
        == 1.0
    )
    assert (
        asyncio.run(run_data.R1FormatVerifier().async_call([], "reason</think><answer>42</answer>", "")).score == 1.0
    )


def test_prepared_mode_never_applies_template(environment, tmp_path):
    prompt = tmp_path / "input.jsonl"
    original = {
        "input": "ALREADY RENDERED",
        "label": "1",
        "metadata": {"verifiers": [{"name": "trusted", "target": "1"}]},
    }
    prompt.write_text(json.dumps(original) + "\n")
    registry = tmp_path / "registry.json"
    registry.write_text('{"trusted":{"factory":"trusted.module.Factory"}}')
    result = prepare(environment, {"prompt_data": str(prompt), "reward_config": str(registry)})
    assert json.loads(Path(result["prompt_data"]).read_text())["input"] == original["input"]
    assert not environment[2].rendered


def test_prepared_mode_rejects_holdout_overlap(environment, tmp_path):
    prompt = tmp_path / "input.jsonl"
    prompt.write_text(
        json.dumps(
            {"input": "same prompt", "label": "1", "metadata": {"verifiers": [{"name": "gsm8k", "target": "1"}]}}
        )
        + "\n"
    )
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"gsm8k": {"factory": run_data.FACTORIES["gsm8k"]}}))
    with pytest.raises(ValueError, match="overlap"):
        prepare(
            environment,
            {"prompt_data": str(prompt), "reward_config": str(registry), "eval_prompt_data": ["heldout", str(prompt)]},
        )


@pytest.mark.parametrize(
    "data",
    [
        {"recipe": "dolci"},
        {"tasks": [{"task": "code", "train_count": 4}]},
        {"tasks": [{"task": "gsm8k", "train_count": True}]},
        {"tasks": [{"task": "gsm8k", "train_count": 2, "prompt_wrapper": "unknown"}]},
    ],
)
def test_unsupported_preparation_fails_without_runtime_imports(data, monkeypatch):
    monkeypatch.setattr(task_verifiers.importlib, "import_module", lambda name: pytest.fail(f"Imported {name}"))
    with pytest.raises(ValueError):
        run_data.validate_data(data)


def test_incomplete_output_is_not_overwritten(environment):
    environment[1].mkdir()
    with pytest.raises(ValueError, match="incomplete"):
        prepare(environment, {"tasks": [{"task": "multiplication", "train_count": 1}]})


def test_modern_if_adapts_target_without_mutation(monkeypatch):
    calls = []

    class Verifier:
        async def async_call(self, tokens, prediction, label, **kwargs):
            calls.append(label)
            return run_data.RewardResult(0.5)

    monkeypatch.setattr(
        task_verifiers.importlib, "import_module", lambda name: SimpleNamespace(IFEvalVerifier=Verifier)
    )
    target = {"instruction_id": ["one"], "kwargs": [{"count": None}]}
    before = copy.deepcopy(target)
    result = asyncio.run(run_data.ManifestIFVerifier().async_call([], "text", target))
    assert result.score == 0.5 and target == before
    assert calls == [repr([target])]


def test_unknown_manifest_verifier_is_rejected(environment, tmp_path):
    path = source_manifest(tmp_path)
    raw = (path.parent / "train.jsonl").read_text().replace('"gsm8k"', '"code"').encode()
    (path.parent / "train.jsonl").write_bytes(raw)
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["train"]["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Unsupported or malformed verifier"):
        prepare(environment, {"rl_manifest": str(path)})
    assert not environment[1].exists()


def test_prompt_token_proof_is_checked(environment, tmp_path):
    prompt = tmp_path / "input.jsonl"
    row = {
        "input": "original",
        "label": "1",
        "metadata": {"prompt_token_ids": [999], "verifiers": [{"name": "gsm8k", "target": "1"}]},
    }
    prompt.write_text(json.dumps(row) + "\n")
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"gsm8k": {"factory": run_data.FACTORIES["gsm8k"]}}))
    with pytest.raises(ValueError, match="token IDs differ"):
        prepare(environment, {"prompt_data": str(prompt), "reward_config": str(registry)})


def test_math_target_punctuation_is_preserved(environment, monkeypatch):
    monkeypatch.setattr(run_data, "_source_rows", lambda name: [{"question": "Find pair", "ground_truth": "(1,2)"}])
    result = prepare(environment, {"tasks": [{"task": "math", "train_count": 1}]})
    assert json.loads(Path(result["prompt_data"]).read_text())["label"] == "(1,2)"


def test_changed_source_during_preparation_never_commits(environment, tmp_path, monkeypatch):
    path = source_manifest(tmp_path)
    original = run_data._verify_row

    def verify(*args):
        original(*args)
        path.write_text(path.read_text() + " ")

    monkeypatch.setattr(run_data, "_verify_row", verify)
    with pytest.raises(ValueError, match="changed while preparing"):
        prepare(environment, {"rl_manifest": str(path)})
    assert not environment[1].exists()


def test_jsonl_unicode_separators_are_content_not_record_boundaries():
    rows = [{"input": "before\u2028middle\u2029after\u0085end"}, {"input": "second"}]
    raw = ("\r\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n").encode()
    assert run_data._rows(raw) == rows


@pytest.mark.parametrize(
    "task,size,train_count,eval_counts",
    [("gsm8k", 100, 20, [0, 10, 80]), ("multiplication", 8100, 128, [0, 20, 2000])],
)
def test_eval_count_does_not_change_training_rows(monkeypatch, task, size, train_count, eval_counts):
    rows = [{"question": f"Question {i}", "ground_truth": str(i)} for i in range(size)]
    monkeypatch.setattr(run_data, "_source_rows", lambda name: rows)
    baseline = None
    for eval_count in eval_counts:
        task_config = {"task": task, "train_count": train_count}
        if eval_count:
            task_config["eval_count"] = eval_count
        data = {"tasks": [task_config]}
        partitions, _, _ = run_data._tasks(data, Tokenizer(), "template", 17)
        if baseline is None:
            baseline = partitions["train"]
        assert partitions["train"] == baseline
        assert len(partitions["eval"]) == eval_count
        assert not {row["input"] for row in baseline} & {row["input"] for row in partitions["eval"]}


def test_named_sample_id_describes_source_split_even_for_heldout(monkeypatch):
    monkeypatch.setattr(run_data, "_source_rows", lambda name: [{"question": "only row", "ground_truth": "1"}])
    for counts, split in [({"train_count": 1}, "train"), ({"eval_count": 1}, "eval")]:
        partitions, _, _ = run_data._tasks({"tasks": [{"task": "gsm8k", **counts}]}, Tokenizer(), "template", 17)
        assert partitions[split][0]["metadata"]["prepared_sample_id"] == "gsm8k:train:0"


def test_existing_named_preparation_never_resamples(environment, monkeypatch):
    data = {"tasks": [{"task": "multiplication", "train_count": 8, "eval_count": 4}]}
    result = prepare(environment, data)
    before = {path.name: path.read_bytes() for path in environment[1].iterdir()}
    monkeypatch.setattr(run_data, "_tasks", lambda *args: pytest.fail("Resampled immutable preparation"))
    assert prepare(environment, data) == result
    assert {path.name: path.read_bytes() for path in environment[1].iterdir()} == before


@pytest.mark.parametrize(
    "keys",
    [
        ("miles",),
        ("artifacts",),
        ("miles", "input_key"),
        ("miles", "chat_template", "path"),
        ("artifacts", "train", "path"),
    ],
)
def test_adopted_manifest_missing_fields_are_input_errors(environment, tmp_path, keys):
    path = source_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    parent = manifest
    for key in keys[:-1]:
        parent = parent[key]
    del parent[keys[-1]]
    path.write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="missing required fields|table/object"):
        prepare(environment, {"rl_manifest": str(path)})


@pytest.mark.parametrize("invalid", ["missing_column", "invalid_metadata", "invalid_template"])
def test_adopted_artifact_errors_have_input_context(environment, tmp_path, invalid):
    path = source_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    if invalid == "invalid_template":
        artifact = manifest["miles"]["chat_template"]
        raw = b"\xff"
        message = "chat template must be UTF-8"
    else:
        artifact = manifest["artifacts"]["train"]
        row = json.loads((path.parent / artifact["path"]).read_text())
        if invalid == "missing_column":
            del row["ground_truth"]
            message = "train row 1.*ground_truth"
        else:
            row["metadata"] = []
            message = "train row 1.*metadata"
        raw = json.dumps(row).encode() + b"\n"
    (path.parent / artifact["path"]).write_bytes(raw)
    artifact["sha256"] = hashlib.sha256(raw).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(InputError, match=message):
        prepare(environment, {"rl_manifest": str(path)})


@pytest.mark.parametrize("invalid", ["json", "fields", "inputs", "missing_output"])
def test_resume_manifest_errors_are_input_errors(environment, invalid):
    data = {"tasks": [{"task": "multiplication", "train_count": 1}]}
    result = prepare(environment, data)
    path = Path(result["manifest"])
    manifest = json.loads(path.read_text())
    if invalid == "json":
        path.write_text("{")
    elif invalid == "missing_output":
        Path(result["prompt_data"]).unlink()
    else:
        if invalid == "fields":
            del manifest["contract"]
        else:
            manifest["inputs"] = []
        path.write_text(json.dumps(manifest))
    with pytest.raises(InputError, match="Invalid JSON|missing required fields|table/object|Cannot read input"):
        prepare(environment, data)


@pytest.mark.parametrize("target", ["{", "null", "[]"])
def test_invalid_ifeval_target_is_an_input_error(environment, monkeypatch, target):
    monkeypatch.setattr(run_data, "_source_rows", lambda name: [{"question": "question", "ground_truth": target}])
    with pytest.raises(InputError, match="ifeval source row 0"):
        prepare(environment, {"tasks": [{"task": "ifeval", "train_count": 1}]})


def test_missing_tokenizer_directory_is_an_input_error(environment):
    hf, output, _ = environment
    with pytest.raises(InputError, match="Cannot read tokenizer directory"):
        run_data.prepare_data(
            {"tasks": [{"task": "multiplication", "train_count": 1}]},
            hf / "missing",
            output,
            max_prompt_length=2000,
            seed=17,
        )
