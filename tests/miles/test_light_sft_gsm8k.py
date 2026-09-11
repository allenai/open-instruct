"""Historical identity and evaluation contracts, without downloading data or weights."""

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
