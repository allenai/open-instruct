"""Keep representative pilot predictions and sampling separate from warmup."""

import copy
import json
from types import SimpleNamespace

import pytest
from scripts.miles import benchmark_aime_capacity as pilot


def test_native_registration_discards_warmup_and_measures_original_outputs(monkeypatch, tmp_path):
    calls = []

    class Engine:
        def generate(self, **kwargs):
            calls.append(copy.deepcopy(kwargs))
            return [{"meta_info": {"completion_tokens": 5, "output_token_logprobs": [1]}}]

        def flush_cache(self):
            calls.append("flush")

        def get_server_info(self):
            return {"internal_states": []}

    modules = {
        "olmo_sglang": SimpleNamespace(register=lambda: calls.append("register")),
        "sglang.srt.entrypoints.engine": SimpleNamespace(Engine=Engine),
    }
    monkeypatch.setattr(pilot.importlib, "import_module", modules.__getitem__)
    monkeypatch.setenv("OLMO_AIME_CAPACITY_TELEMETRY", str(tmp_path))
    pilot.register()
    pilot.register()
    engine = Engine()
    sampling = {"temperature": 1, "max_new_tokens": 16384, "top_p": 0.95}
    for _ in range(2):
        result = engine.generate(input_ids=[[1]], sampling_params=sampling)
        assert result[0]["meta_info"]["output_token_logprobs"] == [1]
    assert sampling["temperature"] == 1
    generations = [call for call in calls if isinstance(call, dict)]
    assert [call["sampling_params"]["temperature"] for call in generations] == [0, 1, 1]
    records = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert [row["phase"] for row in records] == ["warmup", "measured", "measured"]
    assert all("output_token_logprobs" not in row["metadata"][0] for row in records[1:])


def test_command_rejects_changed_draws_before_evaluator_launch(tmp_path):
    receipt = {"tasks": [{"generation": {"num_samples": 16}, "scoring": {"seed": 17}}]}
    with pytest.raises(ValueError, match="sampling protocol"):
        pilot.command(receipt, tmp_path, 128)


def test_capacity_overrides_precede_task_selector(monkeypatch, tmp_path):
    receipt = {
        "tasks": [
            {
                "generation": {
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 16384,
                    "num_samples": 32,
                    "stop_sequences": [],
                },
                "scoring": {"seed": 17},
            }
        ],
        "evaluation": {"provider_kwargs": {"max_running_requests": 128, "random_seed": 17}},
    }
    monkeypatch.setattr(
        pilot.evaluation_runner,
        "command",
        lambda *args: ["olmo-eval", "--harness", "default", "--task", "aime_2024:pass_at_32"],
    )
    argv = pilot.command(receipt, tmp_path, 128)
    assert argv.index("batching.max_batch_sequences=128") < argv.index("--task")
