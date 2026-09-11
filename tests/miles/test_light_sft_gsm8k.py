"""Historical identity and evaluation contracts, without downloading data or weights."""

import asyncio
import copy
import importlib
import json
from types import SimpleNamespace

import pytest
from scripts.miles import launch_light_sft_gsm8k, light_sft_gsm8k


class Tokenizer:
    def encode(self, prompt, add_special_tokens):
        assert add_special_tokens is False
        return [len(prompt), 2, 3]


def request():
    return {
        "native_id": 7,
        "label": "18",
        "doc": {"query": "How many?", "short_answer": "18"},
        "request": {
            "context": "Question: How many?\nAnswer:",
            "generation_kwargs": {
                "max_gen_toks": 512,
                "do_sample": False,
                "temperature": 0,
                "logprobs": 1,
                "num_samples": 1,
                "add_special_tokens": False,
            },
            "stop_sequences": ["Question:", "\n\n"],
            "provider_request": {"endpoint": "/completions"},
        },
    }


def test_historical_checkpoint_rejects_other_architectures():
    config = copy.deepcopy(light_sft_gsm8k.history()["manifest"]["model"]["config_contract"]["architecture"])
    light_sft_gsm8k.validate_geometry(config)
    for name, value in (("num_hidden_layers", 31), ("latent_moe_dim", 512), ("n_routed_experts", 64)):
        changed = {**config, name: value}
        with pytest.raises(ValueError, match="historical 20-layer"):
            light_sft_gsm8k.validate_geometry(changed)


def test_historical_requests_preserve_raw_prompt_and_ids():
    rows, proofs = light_sft_gsm8k.offline_rows([request()], Tokenizer())
    assert rows[0]["input"] == "Question: How many?\nAnswer:"
    assert rows[0]["metadata"]["native_id"] == 7
    assert proofs[0]["prompt_tokens"] == 3


@pytest.mark.parametrize("mutation", ("chat", "sampling", "stop", "label", "duplicate"))
def test_historical_requests_fail_closed(mutation):
    row = request()
    if mutation == "chat":
        row["request"]["context"] = "<user>How many?</user>"
    elif mutation == "sampling":
        row["request"]["generation_kwargs"]["temperature"] = 1
    elif mutation == "stop":
        row["request"]["stop_sequences"] = []
    elif mutation == "label":
        row["label"] = "19"
    with pytest.raises(ValueError):
        light_sft_gsm8k.offline_rows([row, row] if mutation == "duplicate" else [row], Tokenizer())


def test_source_digest_required(tmp_path):
    path = tmp_path / "data.json"
    path.write_text("old")
    expected = light_sft_gsm8k.digest(b"old")
    assert light_sft_gsm8k.checked_bytes(path, expected) == b"old"
    path.write_text("new")
    with pytest.raises(ValueError, match="SHA256"):
        light_sft_gsm8k.checked_bytes(path, expected)


@pytest.mark.parametrize("stage", ("prepare", "core", "audit"))
def test_placement_and_history_mount(stage):
    spec = launch_light_sft_gsm8k.specification("test-image", stage)
    task = spec["tasks"][0]
    assert task["context"]["priority"] == "urgent"
    assert task["constraints"]["cluster"] == ["ai2/holmes" if stage == "core" else "ai2/saturn"]
    assert task["resources"]["gpuCount"] == (4 if stage == "core" else 0)
    if stage == "prepare":
        assert task["datasets"][1]["source"]["beaker"] == "01M13B0CFW65GVAGVWVTDWH4DM"
        assert " run" not in task["arguments"][0]


def test_runtime_configuration_matches_historical_settings(tmp_path):
    pytest.importorskip("miles")
    config = light_sft_gsm8k.configuration(tmp_path)
    config.validate()
    assert config.core.max_sequence_length == 1024
    assert config.core.expert_parallel_size == 2
    assert config.miles["rollout_num_gpus"] == 2
    assert config.miles["rollout_batch_size"] == 8
    assert config.miles["sglang_sampling_backend"] == "flashinfer"
    assert config.miles["sglang_max_total_tokens"] == 771285
    assert config.miles["save_interval"] == 50


def test_full_test_independent_scoring_and_version_checks():
    pytest.importorskip("miles")
    light_sft_eval = importlib.import_module("scripts.miles.light_sft_eval")

    rows, proofs = light_sft_gsm8k.offline_rows([request()], Tokenizer())
    sample = SimpleNamespace(
        metadata=rows[0]["metadata"],
        prompt=rows[0]["input"],
        tokens=[len(rows[0]["input"]), 2, 3, 19],
        response_length=1,
        response="Answer: 18",
        reward=1.0,
        weight_versions=["200"],
        status=SimpleNamespace(name="COMPLETED"),
    )
    report = light_sft_eval.summarize([sample], rows, proofs, 200)
    assert report["correct"] == 1
    sample.weight_versions = ["199"]
    with pytest.raises(ValueError, match="published policy"):
        light_sft_eval.summarize([sample], rows, proofs, 200)
    sample.weight_versions = ["200"]
    sample.tokens[0] += 1
    with pytest.raises(ValueError, match="tokenization"):
        light_sft_eval.summarize([sample], rows, proofs, 200)


def test_native_and_full_test_are_separate_historical_series():
    old = light_sft_gsm8k.history()
    assert old["native_eval"]["temperature"] == 1
    assert old["full_test"]["temperature"] == 0
    assert old["native_eval"]["correct_by_step"]["200"] == 77
    assert old["full_test"]["after_correct"] == 274
    assert json.loads(light_sft_gsm8k.HISTORY_PATH.read_text())["manifest"]["model"]["megatron_iteration"] == "999"


def test_runtime_history_is_inside_docker_copied_configs():
    root = light_sft_gsm8k.HISTORY_PATH.parents[3]
    assert light_sft_gsm8k.HISTORY_PATH.relative_to(root).parts[:2] == ("configs", "miles")
    assert "COPY configs/miles /opt/core-rl/configs/miles" in (root / "runtime/miles/Dockerfile").read_text()


def test_archived_comma_answer_normalization():
    row = request()
    row["label"] = "1450000"
    row["doc"]["short_answer"] = "1,450,000"
    rows, _ = light_sft_gsm8k.offline_rows([row], Tokenizer())
    assert rows[0]["label"] == "1450000"


def test_full_test_rejects_external_snapshot_sampling_state():
    pytest.importorskip("miles")
    module = importlib.import_module("scripts.miles.light_sft_eval")
    worker = object.__new__(module.HistoricalEvaluation)
    for state, hf in ((object(), None), (None, "/snapshot")):
        request = SimpleNamespace(evaluation=True, generate_state=state, hf_dir=hf)
        with pytest.raises(ValueError, match="snapshot generation state"):
            asyncio.run(worker(request))


def test_offline_state_binds_router_after_constructor_and_keeps_sampling(monkeypatch, tmp_path):
    pytest.importorskip("miles")
    module = importlib.import_module("scripts.miles.light_sft_eval")
    types = importlib.import_module("miles.rollout.base_types")
    captured = []
    monkeypatch.setattr(module, "InferenceRolloutFn", lambda value: captured.append(value) or value)
    args = SimpleNamespace(
        prompt_data=str(tmp_path / "train.jsonl"),
        sglang_router_ip=None,
        sglang_router_port=None,
        rollout_stop=None,
        eval_datasets=["native"],
    )
    worker = module.HistoricalEvaluation(types.RolloutFnConstructorInput(args=args, data_source=None))
    assert len(captured) == 1
    with pytest.raises(ValueError, match="live router"):
        worker.offline_function()
    args.sglang_router_ip, args.sglang_router_port = "127.0.0.1", 12345
    offline = worker.offline_function()
    assert offline.args.sglang_router_port == 12345
    assert offline.args.sglang_router_ip == "127.0.0.1"
    assert offline.args.rollout_stop == ["Question:", "\n\n"]
    assert offline.args.eval_datasets[0].temperature == 0
    assert offline.args.eval_datasets[0].max_response_len == 512
    assert args.rollout_stop is None and args.eval_datasets == ["native"]
    args.sglang_router_port = 23456
    assert worker.offline_function().args.sglang_router_port == 23456


def test_retry_stages_full_identical_bytes_and_keeps_source(tmp_path):
    module = importlib.import_module("scripts.miles.light_sft_retry")
    source, destination = tmp_path / "source", tmp_path / "local"
    source.mkdir()
    payload = b"unchanged model bytes" * 1000
    (source / "model.safetensors").write_bytes(payload)
    (source / "config.json").write_text("{}")
    report = module.stage_hf(source, destination)
    assert report["verified_full_payload"]
    assert report["files"]["model.safetensors"]["sha256"] == light_sft_gsm8k.digest(payload)
    assert (destination / "model.safetensors").read_bytes() == payload
    assert (source / "model.safetensors").read_bytes() == payload
    with pytest.raises(FileExistsError):
        module.stage_hf(source, destination)


def test_retry_staging_rejects_insufficient_space(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.miles.light_sft_retry")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="Insufficient"):
        module.stage_hf(source, tmp_path / "local")
    assert not (tmp_path / "local").exists()


def test_retry_preserves_original_outputs_and_stages_before_driver():
    task = launch_light_sft_gsm8k.specification("image", "core-retry")["tasks"][0]
    command = task["arguments"][0]
    assert "20260911-v1-r2" in command
    assert "--source-root " + str(light_sft_gsm8k.ROOT) in command
    assert command.index("light_sft_retry") < command.index("light_sft_gsm8k run")
    assert "--local-hf /tmp/light-sft-hf" in command
    assert "timeout --signal=TERM --kill-after=10s 20m python -m scripts.miles.light_sft_retry" in command
    assert task["resources"]["gpuCount"] == 4
    assert task["constraints"]["cluster"] == ["ai2/holmes"]


def test_native_endpoint_is_retained_if_full_test_fails(tmp_path):
    pytest.importorskip("miles")
    module = importlib.import_module("scripts.miles.light_sft_eval")
    worker = object.__new__(module.HistoricalEvaluation)
    worker.root = tmp_path
    (tmp_path / "core").mkdir()
    sample = SimpleNamespace(
        metadata={"prepared_sample_id": "frozen-id"}, response="18", tokens=[2, 18], weight_versions=["0"]
    )

    async def native(_):
        return SimpleNamespace(data={"gsm8k": {"samples": [sample], "rewards": [1.0], "truncated": [False]}})

    def offline():
        raise RuntimeError("transport failure")

    worker.native = native
    worker.offline_function = offline
    request = SimpleNamespace(evaluation=True, generate_state=None, hf_dir=None, rollout_id=0)
    with pytest.raises(RuntimeError, match="transport failure"):
        asyncio.run(worker(request))
    report = json.loads((tmp_path / "core/native-0.json").read_text())
    assert report["correct"] == 1 and report["samples"][0]["id"] == "frozen-id"


def test_retry_staging_rejects_corrupt_destination(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.miles.light_sft_retry")
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(module.hashlib, "file_digest", lambda *args: SimpleNamespace(hexdigest=lambda: "wrong"))
    with pytest.raises(ValueError, match="differs from the complete source"):
        module.stage_hf(source, tmp_path / "local")


def test_actual_dataset_reader_and_serving_tokenizers_agree(tmp_path):
    pytest.importorskip("miles")
    module = importlib.import_module("scripts.miles.light_sft_tokenization")
    tokenizers = importlib.import_module("tokenizers")
    transformers = importlib.import_module("transformers")
    raw = tokenizers.Tokenizer(tokenizers.models.WordLevel({"[UNK]": 0, "Question:": 1, "12": 2}, unk_token="[UNK]"))
    raw.pre_tokenizer = tokenizers.pre_tokenizers.WhitespaceSplit()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="[UNK]")
    hf = tmp_path / "hf"
    tokenizer.save_pretrained(hf)
    path = tmp_path / "rows.jsonl"
    path.write_text(
        json.dumps({"input": "Question: 12", "label": "12", "metadata": {"prepared_sample_id": "x"}}) + "\n"
    )
    proof = {"prepared_sample_id": "x", "token_ids_sha256": "wrong", "prompt_tokens": 2}
    report = module.compare_partition(path, hf, [proof])
    assert report["valid"]
    assert report["mismatch_counts"] == {"bare_transformers": 0, "checkpoint_json": 0, "sglang": 0, "old_proof": 1}
    assert report["runtime_proofs"][0]["token_ids_sha256"] == light_sft_gsm8k.digest(
        light_sft_gsm8k.json_bytes([1, 2])
    )


def test_tokenization_probe_is_cpu_saturn_and_read_only():
    task = launch_light_sft_gsm8k.specification("image", "tokenization")["tasks"][0]
    assert task["resources"]["gpuCount"] == 0
    assert task["constraints"]["cluster"] == ["ai2/saturn"]
    assert "light_sft_tokenization" in task["arguments"][0]
    assert "light_sft_gsm8k run" not in task["arguments"][0]


def test_existing_corrected_preparation_is_preserved_for_retry(monkeypatch, tmp_path):
    module = importlib.import_module("scripts.miles.light_sft_retry")
    root = tmp_path / "prepared"
    (root / "hf").mkdir(parents=True)
    (root / "hf/model.safetensors").write_bytes(b"weights")
    (root / "preparation.json").write_text('{"corrected_proofs": true}')
    checked = []
    monkeypatch.setattr(module.light_sft_gsm8k, "verify", lambda path: checked.append(path))
    report = module.prepare_retry(tmp_path / "old", root, tmp_path / "local")
    assert checked == [root]
    assert report["prepared_source_root"] == str(root)
    assert (root / "preparation.json").read_text() == '{"corrected_proofs": true}'
    (root / "core").mkdir()
    with pytest.raises(ValueError, match="existing training attempt"):
        module.prepare_retry(tmp_path / "old", root, tmp_path / "local2")
